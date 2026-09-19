import time
import random
import uuid
from typing import Generator
from src.data.schemas import SocialPost

TOPIC_TEMPLATES = {
    "storm_news": [
        "Bão số 4 đang tiến vào đất liền với sức gió cấp 12, cảnh báo sạt lở miền Trung #TinBao #ThienTai",
        "Cập nhật thời tiết: Mưa lớn diện rộng tại Đà Nẵng và Quảng Nam do ảnh hưởng áp thấp #StormNews",
        "Chính quyền khẩn cấp sơ tán dân vùng ven biển tránh bão bùng nổ #TinBao",
        "Các lực lượng cứu hộ sẵn sàng ứng phó bão số 4 tại miền Trung #Baoso4"
    ],
    "ai_tech": [
        "Mô hình AI mới vừa được công bố với khả năng suy luận vượt trội #TechNews #ArtificialIntelligence",
        "OpenAI và Google cạnh tranh nảy lửa trong cuộc đua trí tuệ nhân tạo thế hệ mới #AIChat",
        "Ứng dụng AI vào y học giúp phát hiện bệnh sớm với độ chính xác 98% #HealthTech #AI",
        "Xu hướng tuyển dụng ngành AI và Data Science bùng nổ năm 2026 #AI #Career"
    ],
    "football": [
        "Đội tuyển Việt Nam giành chiến thắng kịch tính ở phút 90+3 #BongDaVietNam #Football",
        "Trận chung kết Cúp C1 diễn ra nảy lửa tối nay, cơ hội chia đều cho hai đội #ChampionsLeague",
        "Huấn luyện viên phát biểu tự tin trước trận đại chiến đêm nay #BongDa",
        "Bàn thắng đẹp mắt từ khoảng cách 30m khiến khán đài bùng nổ #Football"
    ],
    "crypto_market": [
        "Bitcoin vượt mốc giá kỷ lục mới khi dòng tiền quy mô lớn đổ vào thị trường #Crypto #Bitcoin",
        "Thị trường tiền mã hóa biến động mạnh sau thông báo điều chỉnh lãi suất #Fintech #Crypto",
        "Công nghệ Blockchain được áp dụng vào quản lý chuỗi cung ứng toàn cầu #Crypto #Tech"
    ],
    "random_noise": [
        "Hôm nay trời đẹp quá, đi uống cà phê thôi cả nhà #DailyLife",
        "Giao thông giờ cao điểm sài gòn kẹt xe kéo dài #Saigon",
        "Món ăn trưa nay thật ngon miệng #Foodie",
        "Cuối tuần rồi nghỉ ngơi thôi mọi người ơi #Weekend"
    ]
}

class SocialStreamGenerator:
    """Simulates a dynamic real-time social media stream with burst events."""

    def __init__(self, arrival_rate: float = 3.0, burst_prob: float = 0.2):
        self.arrival_rate = arrival_rate
        self.burst_prob = burst_prob
        self.active_burst_topic: str = None
        self.burst_counter = 0

    def generate_post(self) -> SocialPost:
        # Determine topic
        if self.burst_counter > 0 and self.active_burst_topic:
            topic = self.active_burst_topic
            self.burst_counter -= 1
        elif random.random() < self.burst_prob:
            # Trigger a new topic burst!
            topic = random.choice(["storm_news", "ai_tech", "football", "crypto_market"])
            self.active_burst_topic = topic
            self.burst_counter = random.randint(5, 12)  # 5-12 consecutive posts on this topic
        else:
            topic = random.choice(list(TOPIC_TEMPLATES.keys()))

        template = random.choice(TOPIC_TEMPLATES[topic])
        
        # Add subtle variation to text
        post_text = f"{template} (Post #{random.randint(100, 999)})"

        return SocialPost(
            post_id=str(uuid.uuid4())[:8],
            timestamp=time.time(),
            text=post_text,
            author_id=f"user_{random.randint(1000, 9999)}",
            category_hint=topic
        )

    def stream(self, max_posts: int = 100, delay_sec: float = 0.2) -> Generator[SocialPost, None, None]:
        """Yields streaming posts with simulated time delay."""
        for _ in range(max_posts):
            yield self.generate_post()
            time.sleep(delay_sec)
