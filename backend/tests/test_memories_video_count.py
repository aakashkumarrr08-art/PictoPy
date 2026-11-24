"""
Regression tests for memory video count updates upon video deletion.

Validates that deleting videos cascades from memory_videos and that both
the database query layer and the memory API endpoints return live video
and image counts.
"""

import os
import sqlite3
import tempfile
import uuid
from typing import Any, Dict, Iterator, List, cast

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.database.folders import db_create_folders_table
from app.database.images import db_create_images_table, db_delete_images_by_ids
from app.database.memories import (
    MemoryRecord,
    db_create_memories_table,
    db_get_memory,
    db_upsert_memory,
)
from app.database.videos import db_create_videos_table, db_delete_videos_by_ids
from app.database.yolo_mapping import db_create_YOLO_classes_table
from app.routes import memories as memories_route


@pytest.fixture(scope="function")
def test_db(monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    """Point database modules at a fresh isolated tempfile database."""
    db_fd, db_path = tempfile.mkstemp()
    os.close(db_fd)

    monkeypatch.setattr("app.config.settings.DATABASE_PATH", db_path)
    monkeypatch.setattr("app.database.images.DATABASE_PATH", db_path)
    monkeypatch.setattr("app.database.videos.DATABASE_PATH", db_path)
    monkeypatch.setattr("app.database.folders.DATABASE_PATH", db_path)
    monkeypatch.setattr("app.database.yolo_mapping.DATABASE_PATH", db_path)

    db_create_YOLO_classes_table()
    db_create_folders_table()
    db_create_images_table()
    db_create_videos_table()
    db_create_memories_table()

    yield db_path

    if os.path.exists(db_path):
        os.unlink(db_path)


@pytest.fixture
def client() -> Iterator[TestClient]:
    """Test client mounted with memories router."""
    app = FastAPI()
    app.include_router(memories_route.router, prefix="/memories")
    app.state.executor = None
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def media_library(test_db: str) -> Dict[str, List[str]]:
    """Seed sample images and videos belonging to folder-1."""
    conn = sqlite3.connect(test_db)
    conn.execute(
        "INSERT INTO folders (folder_id, folder_path, taggingCompleted) "
        "VALUES ('folder-1', '/library', 1)"
    )

    image_ids = [f"img-{i}" for i in range(3)]
    for i, img_id in enumerate(image_ids):
        conn.execute(
            "INSERT INTO images (id, path, folder_id, thumbnailPath, captured_at, isFavourite) "
            "VALUES (?, ?, 'folder-1', ?, '2024-06-15 10:00:00', 0)",
            (img_id, f"/library/{i}.jpg", f"/thumbs/{i}.jpg"),
        )

    video_ids = [f"vid-{i}" for i in range(3)]
    for i, vid_id in enumerate(video_ids):
        conn.execute(
            "INSERT INTO videos (id, path, folder_id, thumbnailPath, captured_at) "
            "VALUES (?, ?, 'folder-1', ?, '2024-06-15 10:05:00')",
            (vid_id, f"/library/{i}.mp4", f"/thumbs/v{i}.jpg"),
        )

    conn.commit()
    conn.close()
    return {"images": image_ids, "videos": video_ids}


def _make_memory_record(dedupe_key: str, **overrides: Any) -> MemoryRecord:
    record: Dict[str, Any] = {
        "memory_id": str(uuid.uuid4()),
        "dedupe_key": dedupe_key,
        "event_type": "anniversary",
        "status": "complete",
        "title": "Anniversary Moment",
        "surface_date": "2024-06-15",
        "period_start": "2024-06-15 00:00:00",
        "period_end": "2024-06-15 23:59:59",
        "cover_image_id": "img-0",
        "score": 0.88,
    }
    record.update(overrides)
    return cast(MemoryRecord, record)


def test_single_video_deletion_updates_api_count(
    media_library: Dict[str, List[str]], client: TestClient
) -> None:
    """Requirement 1, 2, 3: Memory containing 1 video -> delete video -> API returns video_count = 0."""
    vid = media_library["videos"][0]
    img = media_library["images"][0]
    record = _make_memory_record("single-video-test", image_count=1, video_count=1)
    memory_id = db_upsert_memory(record, [(img, 0, 0.9)], [(vid, 1, 0.8)])
    assert memory_id is not None

    # Verify initial API state
    res_before = client.get(f"/memories/{memory_id}")
    assert res_before.status_code == 200
    story_before = res_before.json()["data"]["memory"]
    assert story_before["video_count"] == 1
    assert len(story_before["videos"]) == 1

    # Delete the video from database
    assert db_delete_videos_by_ids([vid]) is True

    # Verify API returns video_count = 0 and videos = []
    res_after = client.get(f"/memories/{memory_id}")
    assert res_after.status_code == 200
    story_after = res_after.json()["data"]["memory"]
    assert story_after["video_count"] == 0
    assert story_after["videos"] == []

    # Verify list endpoint also returns live video_count = 0
    list_res = client.get("/memories")
    assert list_res.status_code == 200
    cards = list_res.json()["data"]["memories"]
    matching_card = next(c for c in cards if c["memory_id"] == memory_id)
    assert matching_card["video_count"] == 0


def test_multiple_videos_deletion_of_one_updates_count(
    media_library: Dict[str, List[str]], client: TestClient
) -> None:
    """Requirement 4: Memory containing multiple videos, followed by deletion of one."""
    vids = media_library["videos"][:3]
    img = media_library["images"][0]
    record = _make_memory_record("multi-video-test", image_count=1, video_count=3)
    memory_id = db_upsert_memory(
        record,
        [(img, 0, 0.9)],
        [(vids[0], 1, 0.8), (vids[1], 2, 0.7), (vids[2], 3, 0.6)],
    )
    assert memory_id is not None

    # Delete only the first video
    assert db_delete_videos_by_ids([vids[0]]) is True

    # API should now return video_count = 2 and remaining 2 videos
    res = client.get(f"/memories/{memory_id}")
    assert res.status_code == 200
    story = res.json()["data"]["memory"]
    assert story["video_count"] == 2
    assert len(story["videos"]) == 2
    remaining_ids = {v["id"] for v in story["videos"]}
    assert remaining_ids == {vids[1], vids[2]}


def test_mixed_media_preserves_image_count_when_videos_deleted(
    media_library: Dict[str, List[str]], client: TestClient
) -> None:
    """Requirement 5: Memory containing both images and videos, ensuring image counts remain correct."""
    imgs = media_library["images"][:2]
    vids = media_library["videos"][:2]
    record = _make_memory_record("mixed-media-test", image_count=2, video_count=2)
    memory_id = db_upsert_memory(
        record,
        [(imgs[0], 0, 0.9), (imgs[1], 1, 0.85)],
        [(vids[0], 2, 0.75), (vids[1], 3, 0.7)],
    )
    assert memory_id is not None

    # Delete 1 video and 1 image
    assert db_delete_videos_by_ids([vids[0]]) is True
    assert db_delete_images_by_ids([imgs[0]]) is True

    # Check both image_count and video_count are accurately updated independently
    res = client.get(f"/memories/{memory_id}")
    assert res.status_code == 200
    story = res.json()["data"]["memory"]
    assert story["image_count"] == 1
    assert story["video_count"] == 1
    assert len(story["images"]) == 1
    assert len(story["videos"]) == 1
    assert story["images"][0]["id"] == imgs[1]
    assert story["videos"][0]["id"] == vids[1]


def test_memory_with_no_associated_videos(
    media_library: Dict[str, List[str]], client: TestClient
) -> None:
    """Requirement 6: Memory with no associated videos."""
    imgs = media_library["images"][:2]
    record = _make_memory_record("no-video-test", image_count=2, video_count=0)
    memory_id = db_upsert_memory(record, [(imgs[0], 0, 0.9), (imgs[1], 1, 0.85)], [])
    assert memory_id is not None

    res = client.get(f"/memories/{memory_id}")
    assert res.status_code == 200
    story = res.json()["data"]["memory"]
    assert story["video_count"] == 0
    assert story["videos"] == []
    assert story["image_count"] == 2
    assert len(story["images"]) == 2

    # Check DB level query directly
    db_row = db_get_memory(memory_id)
    assert db_row is not None
    assert db_row["live_video_count"] == 0
    assert db_row["live_image_count"] == 2
