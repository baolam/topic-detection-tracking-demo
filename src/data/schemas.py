import time
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class SocialPost(BaseModel):
    """Schema for incoming social media streaming posts."""
    post_id: str
    timestamp: float = Field(default_factory=time.time)
    text: str
    author_id: str = "anon_user"
    hashtags: List[str] = Field(default_factory=list)
    engagement: Dict[str, int] = Field(default_factory=lambda: {"likes": 0, "shares": 0})
    category_hint: Optional[str] = None  # Used for ground-truth synthetic evaluation

class MicroClusterSummary(BaseModel):
    """Summary representation of a Micro-cluster state."""
    cluster_id: str
    weight: float
    center: List[float]
    radius: float
    creation_time: float
    last_update_time: float
    is_potential: bool

class TrendingTopic(BaseModel):
    """Macro-cluster / Active Trending Topic representation."""
    topic_id: str
    label: str
    keywords: List[str]
    sample_posts: List[str]
    total_weight: float
    micro_cluster_count: int
    last_updated: float
    is_bursting: bool = False
    burst_score: float = 0.0

