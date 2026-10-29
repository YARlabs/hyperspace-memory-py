from __future__ import annotations

import hashlib
import math
import os
import time
import uuid
from typing import Any, Dict, List, Optional, Union

from hyperspace import HyperspaceClient


def deterministic_numeric_id(memory_id_str: str) -> int:
    """
    Derives a deterministic 31-bit positive integer ID from a string identifier
    using BLAKE2b (immune to PYTHONHASHSEED randomization across processes).
    """
    digest = hashlib.blake2b(memory_id_str.encode("utf-8"), digest_size=8).digest()
    val = int.from_bytes(digest, byteorder="big", signed=False) % 2147483646
    return val + 1


def mrl_truncate_and_normalize(v: List[float]) -> List[float]:
    """MRL truncation to 129D + upper-sheet Lorentz normalization (33D Head + 96D Tail)."""
    if not v:
        return []
    truncated = list(v[:129]) if len(v) > 129 else list(v)
    if len(truncated) < 33:
        return truncated

    # Lorentz part (first 33 elements: 1 timelike x_0 + 32 spatial) — enforce upper-sheet constraint
    lorentz = truncated[:33]
    spatial_norm_sq = sum(x * x for x in lorentz[1:])
    lorentz[0] = math.sqrt(1.0 + spatial_norm_sq)

    # Euclidean part (remaining elements) — L2 normalize
    euclidean = truncated[33:]
    euc_norm_sq = sum(x * x for x in euclidean)
    euc_norm = math.sqrt(euc_norm_sq)
    if euc_norm > 1e-12:
        euclidean = [x / euc_norm for x in euclidean]

    return lorentz + euclidean


def frechet_mean_hyperboloid(vectors: List[List[float]]) -> List[float]:
    """
    Computes the centroid / Fréchet mean on the Product Manifold (33D Lorentz Head + Euclidean Tail).
    """
    if not vectors:
        return []
    dim = len(vectors[0])
    if dim == 0:
        return []
    n = float(len(vectors))
    mean = [0.0] * dim
    for vec in vectors:
        limit = min(dim, len(vec))
        for i in range(limit):
            mean[i] += vec[i]
    mean = [x / n for x in mean]

    if dim >= 33:
        spatial_sq = sum(x * x for x in mean[1:33])
        mean[0] = math.sqrt(1.0 + spatial_sq)
        tail_sq = sum(x * x for x in mean[33:])
        tail_norm = math.sqrt(tail_sq)
        if tail_norm > 1e-12:
            for i in range(33, dim):
                mean[i] /= tail_norm
    return mean


class Memory:
    """
    Drop-in replacement for Mem0, Zep, and MemGPT Memory powered by HyperspaceDB v4.2.

    Features:
    - 801D Product-Manifold Hybrid embeddings (33D Lorentz Head + 768D Euclidean Tail)
      with 129D in-RAM MRL cascades.
    - Deterministic cross-process ID hashing (BLAKE2b) and O(1) point lookups.
    - Native server-side filter pushdown for multi-tenant isolation (user_id, agent_id, session_id).
    - Autonomous conflict resolution via directed `SUPERSEDES` edges in the CSR KAG graph.
    - Geometric hallucination verification via the HyperspaceDB Tribunal Router.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config = config or {}
        vs_cfg = config.get("vector_store", {}).get("config", {})
        self.host: str = (
            config.get("host")
            or vs_cfg.get("host")
            or os.environ.get("HYPERSPACE_HOST")
            or "localhost:50051"
        )
        self.api_key: str = (
            config.get("api_key")
            or vs_cfg.get("api_key")
            or os.environ.get("HYPERSPACE_API_KEY")
            or os.environ.get("YAR_API_KEY")
            or os.environ.get("CDE_API_KEY")
            or "I_LOVE_HYPERSPACEDB"
        )
        self.collection: str = config.get("collection_name") or vs_cfg.get("collection_name") or "agent_memories"
        self.quantization: str = config.get("quantization") or "extreme"
        self.encryption_key: Optional[str] = (
            config.get("encryption_key")
            or config.get("encryptionKey")
            or config.get("password")
            or os.environ.get("HYPERSPACE_ENCRYPTION_KEY")
        )
        self.decay_lambda: float = float(config.get("decay_lambda", 0.005))
        self.auto_supersede: bool = bool(config.get("auto_supersede", True))
        self.dimension: int = int(config.get("dimension", 801))
        self.metric: str = str(config.get("metric", "hybrid"))

        is_local = self.host.startswith("localhost") or self.host.startswith("127.0.0.1")
        grpc_key = (
            "I_LOVE_HYPERSPACEDB"
            if (is_local and self.api_key.startswith("sk_"))
            else (self.api_key or "I_LOVE_HYPERSPACEDB")
        )
        self.client: HyperspaceClient = config.get("client") or HyperspaceClient(self.host, grpc_key)
        self._owns_client: bool = "client" not in config
        self._collection_ensured: bool = False
        self._ensure_collection()

    def close(self) -> None:
        """Closes the underlying gRPC connection pool if owned by this Memory instance."""
        if self._owns_client and hasattr(self.client, "close"):
            self.client.close()

    def __enter__(self) -> "Memory":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def _ensure_collection(self, force: bool = False) -> None:
        """Creates default memory collection once if it does not exist."""
        if self._collection_ensured and not force:
            return
        cutoff_dim = 129 if self.dimension >= 129 else self.dimension
        schema = {
            "components": [
                {
                    "name": "default",
                    "metric": self.metric,
                    "fullDimension": self.dimension,
                    "weight": 1.0,
                }
            ],
            "cascadePipeline": (
                [
                    {
                        "componentName": "default",
                        "cutoffDimension": cutoff_dim,
                        "storeInRam": True,
                        "rerankTopK": 50,
                    }
                ]
                if self.dimension > cutoff_dim
                else []
            ),
            "encrypted": bool(self.encryption_key),
        }
        if self.encryption_key:
            self.client.register_collection_key(
                self.collection, self.encryption_key, metric=self.metric
            )

        try:
            self.client.create_collection(
                self.collection,
                schema,
                quantization=self.quantization,
                encryption_key=self.encryption_key,
                encrypted=bool(self.encryption_key),
            )
            self._collection_ensured = True
        except Exception:
            self._collection_ensured = True

    def _generate_numeric_id(self, memory_id_str: str) -> int:
        return deterministic_numeric_id(memory_id_str)

    @staticmethod
    def _build_server_filter(
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        session_id: Optional[str] = None,
        extra_filters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, str]:
        server_filter: Dict[str, str] = {}
        if user_id:
            server_filter["user_id"] = str(user_id)
        if agent_id:
            server_filter["agent_id"] = str(agent_id)
        if run_id:
            server_filter["run_id"] = str(run_id)
        if session_id:
            server_filter["session_id"] = str(session_id)
        if extra_filters:
            for k, v in extra_filters.items():
                if isinstance(v, (str, int, float, bool)):
                    server_filter[str(k)] = str(v)
        return server_filter

    def add(
        self,
        messages: Union[str, List[Dict[str, Any]]],
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        session_id: Optional[str] = None,
        importance: Optional[float] = None,
        tags: Optional[List[str]] = None,
        memory_type: Optional[str] = "episodic",
        metadata: Optional[Dict[str, Any]] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Stores a new memory item with autonomous conflict resolution (Temporal Superseding)
        and Zero-Knowledge protection.
        """
        if isinstance(messages, list):
            text = " ".join(
                [m.get("content", str(m)) if isinstance(m, dict) else str(m) for m in messages]
            )
        else:
            text = str(messages)

        memory_id = str(uuid.uuid4())
        num_id = self._generate_numeric_id(memory_id)
        now_ts = time.time()

        meta = dict(metadata or {})
        if filters:
            for k, v in filters.items():
                meta.setdefault(k, v)
        if user_id:
            meta["user_id"] = user_id
        if agent_id:
            meta["agent_id"] = agent_id
        if run_id:
            meta["run_id"] = run_id
        if session_id:
            meta["session_id"] = session_id
        if importance is not None:
            meta["importance"] = str(importance)
        if tags:
            meta["tags"] = ",".join(tags)
        if memory_type:
            meta["memory_type"] = memory_type

        meta["memory_id"] = memory_id
        meta["text"] = text
        meta["timestamp"] = str(now_ts)
        meta["created_at"] = str(now_ts)
        meta["last_accessed_at"] = str(now_ts)
        meta["access_count"] = "0"
        meta["is_superseded"] = "false"
        meta["stored_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now_ts))

        # Autonomous conflict resolution before insertion
        superseded_ids: List[str] = []
        edges_to_link: List[Dict[str, Any]] = []
        if self.auto_supersede and len(text) > 10:
            try:
                candidates = self.search(
                    text, user_id=user_id, session_id=session_id, limit=3
                )
                for cand in candidates:
                    cand_id = cand.get("id")
                    cand_score = float(cand.get("score", 0.0))
                    if cand_id and cand_id != memory_id and cand_score >= 0.82:
                        cand_num_id = self._generate_numeric_id(cand_id)
                        edges_to_link.append(
                            {
                                "source_id": num_id,
                                "target_id": cand_num_id,
                                "relation": "SUPERSEDES",
                                "weight": cand_score,
                            }
                        )
                        superseded_ids.append(cand_id)
                        cand_meta = dict(cand.get("metadata") or {})
                        cand_meta["is_superseded"] = "true"
                        cand_meta["superseded_by"] = memory_id
                        try:
                            self.client.update_payload(
                                cand_num_id,
                                {k: str(v) for k, v in cand_meta.items()},
                                collection=self.collection,
                            )
                        except Exception:
                            pass
            except Exception:
                pass

        vector = self.client.vectorize(text, metric=self.metric)
        string_meta = {k: str(v) for k, v in meta.items()}
        self.client.insert(
            num_id, vector=vector, metadata=string_meta, collection=self.collection
        )

        if edges_to_link:
            self.link_batch(edges_to_link)

        return {
            "results": [
                {
                    "id": memory_id,
                    "event": "ADD",
                    "data": text,
                    "metadata": string_meta,
                    "superseded_memories": superseded_ids,
                }
            ]
        }

    def add_batch(
        self,
        items: List[Dict[str, Any]],
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        High-throughput batch memory insertion using a single gRPC `batch_insert` call.
        """
        if not items:
            return {"results": []}

        now_ts = time.time()
        stored_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now_ts))
        vectors: List[List[float]] = []
        ids: List[int] = []
        metadatas: List[Dict[str, str]] = []
        results_out: List[Dict[str, Any]] = []

        for item in items:
            text = str(item.get("text") or item.get("messages") or item.get("data") or "")
            if not text:
                continue
            memory_id = str(item.get("id") or uuid.uuid4())
            num_id = self._generate_numeric_id(memory_id)
            meta = dict(item.get("metadata") or {})
            uid = item.get("user_id") or user_id
            aid = item.get("agent_id") or agent_id
            sid = item.get("session_id") or session_id
            if uid:
                meta["user_id"] = str(uid)
            if aid:
                meta["agent_id"] = str(aid)
            if sid:
                meta["session_id"] = str(sid)
            if "tags" in item and item["tags"]:
                meta["tags"] = (
                    ",".join(item["tags"])
                    if isinstance(item["tags"], list)
                    else str(item["tags"])
                )
            meta["memory_id"] = memory_id
            meta["text"] = text
            meta["timestamp"] = str(now_ts)
            meta["created_at"] = str(now_ts)
            meta["last_accessed_at"] = str(now_ts)
            meta["access_count"] = "0"
            meta["is_superseded"] = "false"
            meta["stored_at"] = stored_at

            vec = item.get("vector") or self.client.vectorize(text, metric=self.metric)
            str_meta = {k: str(v) for k, v in meta.items()}
            vectors.append(vec)
            ids.append(num_id)
            metadatas.append(str_meta)
            results_out.append(
                {
                    "id": memory_id,
                    "event": "ADD",
                    "data": text,
                    "metadata": str_meta,
                }
            )

        if vectors:
            self.client.batch_insert(
                vectors=vectors,
                ids=ids,
                metadatas=metadatas,
                collection=self.collection,
            )

        return {"results": results_out}

    def search(
        self,
        query: str,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        session_id: Optional[str] = None,
        min_score: Optional[float] = None,
        tags: Optional[List[str]] = None,
        limit: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        apply_decay: bool = True,
        use_wave: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Searches memories with server-side filter pushdown, Ebbinghaus Temporal Decay,
        Access Frequency Boost, and Superseded Fact Penalization.
        """
        query_vector = self.client.vectorize(query, metric=self.metric)
        candidate_limit = max(limit * 4, 25)
        server_filter = self._build_server_filter(
            user_id=user_id,
            agent_id=agent_id,
            run_id=run_id,
            session_id=session_id,
            extra_filters=filters,
        )

        results = self.client.search(
            query_vector,
            top_k=candidate_limit,
            filter=server_filter if server_filter else None,
            collection=self.collection,
            use_wave=use_wave,
        )

        now_ts = time.time()
        memories: List[Dict[str, Any]] = []
        for match in results:
            if isinstance(match, dict):
                match_meta = match.get("metadata", {}) or {}
                match_id = match.get("id")
                match_dist = float(match.get("distance", 0.0))
                match_score = 1.0 / (1.0 + max(0.0, match_dist))
            else:
                match_meta = getattr(match, "metadata", {}) or {}
                match_id = getattr(match, "id", "")
                match_score = float(getattr(match, "score", 0.95))

            if user_id and match_meta.get("user_id") != str(user_id):
                continue
            if agent_id and match_meta.get("agent_id") != str(agent_id):
                continue
            if run_id and match_meta.get("run_id") != str(run_id):
                continue
            if session_id and match_meta.get("session_id") != str(session_id):
                continue
            if tags:
                mem_tags = set((match_meta.get("tags") or "").split(","))
                if not any(t in mem_tags for t in tags):
                    continue

            if apply_decay:
                stored_ts = float(
                    match_meta.get("last_accessed_at")
                    or match_meta.get("timestamp")
                    or now_ts
                )
                hours_elapsed = max(0.0, (now_ts - stored_ts) / 3600.0)
                recency_decay = math.exp(-self.decay_lambda * hours_elapsed)
                access_count = int(match_meta.get("access_count", "0"))
                access_boost = 1.0 + 0.05 * math.log(1.0 + access_count)
                imp = float(match_meta.get("importance", 5.0))
                importance_boost = 1.0 + 0.05 * (imp - 5.0)
                is_superseded = match_meta.get("is_superseded") == "true"
                supersede_penalty = 0.20 if is_superseded else 1.0
                final_score = (
                    match_score
                    * recency_decay
                    * access_boost
                    * importance_boost
                    * supersede_penalty
                )
            else:
                final_score = match_score

            if min_score is not None and final_score < min_score:
                continue

            memories.append(
                {
                    "id": match_meta.get("memory_id", str(match_id)),
                    "memory": match_meta.get("text", query),
                    "score": round(final_score, 4),
                    "raw_similarity": round(match_score, 4),
                    "user_id": match_meta.get("user_id", user_id),
                    "agent_id": match_meta.get("agent_id", agent_id),
                    "run_id": match_meta.get("run_id", run_id),
                    "session_id": match_meta.get("session_id", session_id),
                    "tags": (
                        match_meta.get("tags", "").split(",")
                        if match_meta.get("tags")
                        else None
                    ),
                    "importance": (
                        float(match_meta.get("importance", 5.0))
                        if match_meta.get("importance")
                        else None
                    ),
                    "is_superseded": match_meta.get("is_superseded") == "true",
                    "metadata": match_meta,
                }
            )

        memories.sort(key=lambda x: x["score"], reverse=True)
        return memories[:limit]

    def get_all(
        self,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        session_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Retrieve all stored memories for a user/agent using scroll with filter fallback."""
        filters_list: List[Dict[str, Any]] = []
        if user_id:
            filters_list.append({"type": "match", "key": "user_id", "value": str(user_id)})
        if agent_id:
            filters_list.append({"type": "match", "key": "agent_id", "value": str(agent_id)})
        if run_id:
            filters_list.append({"type": "match", "key": "run_id", "value": str(run_id)})
        if session_id:
            filters_list.append({"type": "match", "key": "session_id", "value": str(session_id)})

        try:
            pts = self.client.scroll(
                limit=limit,
                offset=0,
                filters=filters_list if filters_list else None,
                collection=self.collection,
            )
            if pts:
                out: List[Dict[str, Any]] = []
                for p in pts:
                    m = p.get("metadata") or {}
                    out.append(
                        {
                            "id": m.get("memory_id", str(p.get("id"))),
                            "memory": m.get("text", ""),
                            "score": 1.0,
                            "user_id": m.get("user_id", user_id),
                            "agent_id": m.get("agent_id", agent_id),
                            "run_id": m.get("run_id", run_id),
                            "session_id": m.get("session_id", session_id),
                            "tags": m.get("tags", "").split(",") if m.get("tags") else None,
                            "importance": float(m.get("importance", 5.0)) if m.get("importance") else None,
                            "is_superseded": m.get("is_superseded") == "true",
                            "metadata": m,
                        }
                    )
                return out
        except Exception:
            pass

        return self.search(
            "*",
            user_id=user_id,
            agent_id=agent_id,
            run_id=run_id,
            session_id=session_id,
            limit=limit,
            apply_decay=False,
        )

    def get(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a specific memory by ID in O(1) via `get_points`."""
        num_id = self._generate_numeric_id(memory_id)
        try:
            pts = self.client.get_points([num_id], collection=self.collection)
            if pts:
                p = pts[0]
                m = p.get("metadata") or {}
                return {
                    "id": m.get("memory_id", memory_id),
                    "memory": m.get("text", ""),
                    "score": 1.0,
                    "user_id": m.get("user_id"),
                    "agent_id": m.get("agent_id"),
                    "run_id": m.get("run_id"),
                    "session_id": m.get("session_id"),
                    "tags": m.get("tags", "").split(",") if m.get("tags") else None,
                    "importance": float(m.get("importance", 5.0)) if m.get("importance") else None,
                    "is_superseded": m.get("is_superseded") == "true",
                    "metadata": m,
                }
        except Exception:
            pass
        return None

    def update(self, memory_id: str, data: str) -> Dict[str, Any]:
        """Update an existing memory item while preserving existing metadata."""
        num_id = self._generate_numeric_id(memory_id)
        existing = self.get(memory_id)
        meta: Dict[str, str] = {}
        if existing and isinstance(existing.get("metadata"), dict):
            meta.update({k: str(v) for k, v in existing["metadata"].items()})

        try:
            self.client.delete(num_id, collection=self.collection)
        except Exception:
            pass

        vector = self.client.vectorize(data, metric=self.metric)
        meta["memory_id"] = memory_id
        meta["text"] = data
        meta["updated_at"] = str(time.time())
        self.client.insert(
            num_id, vector=vector, metadata=meta, collection=self.collection
        )

        return {
            "message": "Memory updated successfully",
            "id": memory_id,
            "data": data,
        }

    def delete(self, memory_id: str) -> Dict[str, Any]:
        """Delete a single memory item by ID."""
        num_id = self._generate_numeric_id(memory_id)
        self.client.delete(num_id, collection=self.collection)
        return {"message": f"Memory {memory_id} deleted successfully"}

    def delete_all(
        self,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Delete all memories matching criteria."""
        memories = self.get_all(
            user_id=user_id,
            agent_id=agent_id,
            run_id=run_id,
            session_id=session_id,
            limit=1000,
        )
        count = 0
        for mem in memories:
            if mem.get("id"):
                self.delete(mem["id"])
                count += 1
        return {"message": f"Deleted {count} memories"}

    def reset(self) -> Dict[str, Any]:
        """Purge and reset the memory collection."""
        try:
            self.client.delete_collection(self.collection)
        except Exception:
            pass
        self._collection_ensured = False
        self._ensure_collection(force=True)
        return {"message": "Memory collection reset successfully"}

    def history(self, memory_id: str) -> List[Dict[str, Any]]:
        """Retrieve modification history for a memory item."""
        mem = self.get(memory_id)
        if not mem:
            return []
        return [
            {
                "id": memory_id,
                "event": "ADD",
                "data": mem.get("memory"),
                "timestamp": mem.get("metadata", {}).get("timestamp"),
            }
        ]

    def list_sessions(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Enumerate active conversation sessions discovered across memories."""
        all_mems = self.get_all(limit=limit)
        session_map: Dict[str, Dict[str, Any]] = {}
        for h in all_mems:
            sid = h.get("session_id")
            if sid:
                if sid not in session_map:
                    session_map[sid] = {
                        "session_id": sid,
                        "observed_count": 1,
                        "sample": (h.get("memory") or "")[:80],
                    }
                else:
                    session_map[sid]["observed_count"] += 1
        return list(session_map.values())

    def consolidate(
        self,
        topic_query: str,
        limit: int = 10,
        summary_text: Optional[str] = None,
        persist: bool = True,
        archive_sources: bool = False,
    ) -> Dict[str, Any]:
        """
        Consolidate a cluster of related episodic memories into a single abstract semantic concept
        using the Fréchet Mean on the Product Manifold.
        """
        hits = self.search(topic_query, limit=limit)
        if not hits:
            raise ValueError(
                f"No matching memories found to consolidate for: '{topic_query}'"
            )

        ids = [self._generate_numeric_id(h["id"]) for h in hits]
        try:
            points = self.client.get_points(ids, collection=self.collection)
            vectors = [p.get("vector") for p in points if p.get("vector")]
        except Exception:
            vectors = [
                self.client.vectorize(h["memory"], metric=self.metric) for h in hits
            ]

        if not vectors:
            raise ValueError("No valid vectors retrieved for consolidation.")

        mean_vector = frechet_mean_hyperboloid(vectors)
        summary = (
            summary_text
            or f"[Consolidated] {topic_query}: synthesized from {len(vectors)} memories"
        )
        consolidated_id = None

        if persist:
            consolidated_id = f"consolidated_{int(time.time())}_{uuid.uuid4().hex[:6]}"
            num_id = self._generate_numeric_id(consolidated_id)
            metadata = {
                "memory_id": consolidated_id,
                "text": summary,
                "session_id": "consolidated_global",
                "tags": f"consolidated,{topic_query}",
                "memory_type": "semantic",
                "importance": "8",
                "source_count": str(len(vectors)),
                "source_ids": ",".join(h["id"] for h in hits),
                "stored_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            self.client.insert(
                num_id,
                vector=mean_vector,
                metadata=metadata,
                collection=self.collection,
            )
            edges = [
                {
                    "source_id": num_id,
                    "target_id": src_id,
                    "relation": "CONTAINS",
                    "weight": 1.0,
                }
                for src_id in ids
            ]
            self.link_batch(edges)

        if archive_sources:
            for h in hits:
                try:
                    self.delete(h["id"])
                except Exception:
                    pass

        return {
            "status": "consolidated",
            "topic": topic_query,
            "consolidated_id": consolidated_id,
            "summary": summary,
            "source_count": len(vectors),
            "source_ids": [h["id"] for h in hits],
            "persisted": persist,
            "archived_sources": archive_sources,
            "consolidated_dimension": len(mean_vector),
            "frechet_mean_preview": (
                mean_vector[:6] if len(mean_vector) >= 6 else mean_vector
            ),
        }

    def verify_claim(
        self, premise: str, conclusion: str, threshold: float = 0.30
    ) -> Dict[str, Any]:
        """
        Verify whether a logical claim is geometrically consistent with a premise in 129D MRL hybrid space
        (33D Lorentz + 96D Cosine) to block hallucinations.
        """
        u_raw = self.client.vectorize(premise, metric=self.metric)
        v_raw = self.client.vectorize(conclusion, metric=self.metric)

        u = mrl_truncate_and_normalize(u_raw)
        v = mrl_truncate_and_normalize(v_raw)

        u_l = u[:33]
        v_l = v[:33]
        prod = -u_l[0] * v_l[0] + sum(a * b for a, b in zip(u_l[1:], v_l[1:]))
        lorentz_dist = math.acosh(max(-prod, 1.0))

        u_e = u[33:]
        v_e = v[33:]
        dot = sum(a * b for a, b in zip(u_e, v_e))
        norm_u = math.sqrt(sum(a * a for a in u_e))
        norm_v = math.sqrt(sum(b * b for b in v_e))
        cosine_dist = 1.0 - dot / (norm_u * norm_v + 1e-9)

        dist = lorentz_dist + cosine_dist
        trust_score = 1.0 / (1.0 + dist)
        verified = trust_score > threshold

        return {
            "status": "VERIFIED" if verified else "REJECTED",
            "trust_score": round(trust_score, 4),
            "threshold": threshold,
            "lorentz_distance": round(lorentz_dist, 4),
            "cosine_distance": round(cosine_dist, 4),
            "total_distance": round(dist, 4),
            "reason": (
                "Claim is geometrically consistent with premise in Product-Manifold space."
                if verified
                else f"Geodesic violation: distance {dist:.4f} exceeds threshold."
            ),
            "vector_dimension": len(u),
        }

    def verify_generation(
        self,
        llm_answer: str,
        context_memory_ids: List[str],
        max_distance_threshold: Optional[float] = None,
        check_geodesic: bool = True,
    ) -> Dict[str, Any]:
        """
        Verifies an LLM response against retrieved source memory IDs using the server-side
        HyperspaceDB Tribunal Router (`verify_generation` RPC).
        """
        numeric_str_ids = [
            str(int(cid)) if cid.isdigit() else str(self._generate_numeric_id(cid))
            for cid in context_memory_ids
        ]
        res = self.client.verify_generation(
            llm_answer=llm_answer,
            context_ids=numeric_str_ids,
            collection=self.collection,
            max_distance_threshold=max_distance_threshold,
            check_geodesic=check_geodesic,
        )
        return {
            "trust_score": res.trust_score,
            "is_hallucination": res.is_hallucination,
            "is_valid": res.is_valid,
            "min_distance": res.min_distance,
            "avg_distance": res.avg_distance,
            "geodesic_distance": res.geodesic_distance,
            "geodesic_hops": res.geodesic_hops,
            "cone_satisfied": res.cone_satisfied,
            "details": res.details,
            "resolved_context_ids": res.resolved_context_ids,
        }

    def explore_hierarchy(
        self, concept_id: int, direction: str = "down", limit: int = 32
    ) -> Any:
        """Traverse concept taxonomy in Lorentz space."""
        if direction == "up":
            return self.client.get_concept_parents(
                concept_id, layer=0, limit=limit, collection=self.collection
            )
        return self.client.get_subsumption_tree(
            concept_id, max_depth=3, collection=self.collection
        )

    def link(
        self,
        source_id: int,
        target_id: int,
        relation: str = "OWNS",
        weight: float = 1.0,
    ) -> bool:
        """Create explicit directed edge between memories or concepts in the native KAG graph."""
        try:
            return bool(
                self.client.link(
                    source_id,
                    target_id,
                    relation=relation,
                    weight=weight,
                    collection=self.collection,
                )
            )
        except Exception:
            return False

    def link_batch(self, edges: List[Dict[str, Any]]) -> int:
        """Batch-insert explicit directed edges between memories/entities in CSR topology."""
        if not edges:
            return 0
        try:
            return int(self.client.link_batch(edges, collection=self.collection))
        except Exception:
            return 0

    def unlink(
        self, source_id: int, target_id: int, relation: str = "OWNS"
    ) -> bool:
        """Delete an explicit directed edge with zero-lag invalidation."""
        try:
            return bool(
                self.client.unlink(
                    source_id,
                    target_id,
                    relation=relation,
                    collection=self.collection,
                )
            )
        except Exception:
            return False

    def induce_edges(self, rules: List[Dict[str, Any]]) -> int:
        """Induce topological edges from metadata properties across memory entries."""
        if not rules:
            return 0
        try:
            return int(
                self.client.induce_edges_from_metadata(
                    rules, collection=self.collection
                )
            )
        except Exception:
            return 0

    def auto_reconsolidate(
        self,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        min_cluster_size: int = 2,
    ) -> List[Dict[str, Any]]:
        """
        Background autonomous sleep/reconsolidation worker.
        Groups uncompressed episodic memories by topic, computes Fréchet Mean on the Product Manifold,
        persists semantic concepts, and batch-links them via KAG graph `CONTAINS` edges.
        """
        memories = self.get_all(user_id=user_id, session_id=session_id, limit=100)
        episodic = [
            m
            for m in memories
            if m.get("metadata", {}).get("memory_type", "episodic") == "episodic"
            and not m.get("is_superseded")
        ]
        if len(episodic) < min_cluster_size:
            return []

        groups: Dict[str, List[Dict[str, Any]]] = {}
        for mem in episodic:
            tags = mem.get("tags") or ["general"]
            primary_tag = tags[0] if tags else "general"
            groups.setdefault(primary_tag, []).append(mem)

        results: List[Dict[str, Any]] = []
        for tag, cluster in groups.items():
            if len(cluster) >= min_cluster_size:
                try:
                    res = self.consolidate(
                        topic_query=tag,
                        limit=len(cluster),
                        summary_text=(
                            f"[AutoReconsolidated] Semantic concept for '{tag}': "
                            f"synthesized from {len(cluster)} episodes"
                        ),
                        persist=True,
                        archive_sources=False,
                    )
                    results.append(res)
                except Exception:
                    pass
        return results

    def stats(self) -> Dict[str, Any]:
        """Get operational statistics and health diagnostics for the memory collection."""
        try:
            st = self.client.get_collection_stats(self.collection)
            count = st.get("count", 0)
            encrypted = st.get("encrypted", bool(self.encryption_key))
        except Exception:
            count = 0
            encrypted = bool(self.encryption_key)

        return {
            "collection": self.collection,
            "host": self.host,
            "status": "ready",
            "approx_vector_count": count,
            "encrypted": encrypted,
            "schema": {
                "metric": self.metric,
                "fullDimension": self.dimension,
                "ramMrlDimension": 129 if self.dimension >= 129 else self.dimension,
            },
        }


# Backwards-compatible alias
HyperspaceMemory = Memory
