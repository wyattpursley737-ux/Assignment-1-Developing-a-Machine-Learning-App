from dataclasses import dataclass

@dataclass(frozen=True)
class Settings:
    app_name: str = "CommodityCast"
    commodity_name: str = "Gold Futures"
    ticker: str = "GC=F"

    start_date: str = "2010-01-01"
    end_date: str = "2025-12-31"
    interval: str = "1d"

    horizon_days: int = 5  # predict close price h trading days ahead

    cache_dir: str = "data_cache"
    artifacts_dir: str = "artifacts"
    outputs_dir: str = "outputs"

    test_size: float = 0.2  # last 20% reserved for test (chronological)

    # Report header fields (edit these to match your class)
    author_name: str = "Jeff Pursley"
    course_name: str = "ML 305: Applied Machine Learning"
    instructor_name: str = "Instructor Name Here"

settings = Settings()
