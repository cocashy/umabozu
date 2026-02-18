from datetime import date

from umabozu.data_collection.scraper import ScraperClient


def test_fetch_and_save_shutsubahyou_html_writes_file(tmp_path):
    html = "<html>Shutsubahyou HTML</html>"
    race_id = "123456789012"
    save_dir = tmp_path / "upcoming"

    def fetch_html(_url: str, _encoding: str) -> str | None:
        return html

    client = ScraperClient(fetch_html=fetch_html, wait_fn=lambda: None)
    result = client.fetch_and_save_shutsubahyou_html(race_id, str(save_dir))

    expected_path = save_dir / f"{race_id}.html"
    assert result == str(expected_path)
    assert expected_path.read_text(encoding="utf-8") == html


def test_fetch_and_save_shutsubahyou_html_uses_cache(tmp_path):
    race_id = "123456789012"
    save_dir = tmp_path / "upcoming"
    save_dir.mkdir(parents=True, exist_ok=True)
    expected_path = save_dir / f"{race_id}.html"
    expected_path.write_text("cached", encoding="utf-8")

    def fetch_html(_url: str, _encoding: str) -> str | None:
        raise AssertionError("network should not be called")

    client = ScraperClient(fetch_html=fetch_html, wait_fn=lambda: None)
    result = client.fetch_and_save_shutsubahyou_html(race_id, str(save_dir))

    assert result == str(expected_path)
    assert expected_path.read_text(encoding="utf-8") == "cached"


def test_fetch_and_save_shutsubahyou_html_returns_none_on_failure(tmp_path):
    race_id = "123456789012"
    save_dir = tmp_path / "upcoming"

    def fetch_html(_url: str, _encoding: str) -> str | None:
        return None

    client = ScraperClient(fetch_html=fetch_html, wait_fn=lambda: None)
    result = client.fetch_and_save_shutsubahyou_html(race_id, str(save_dir))

    assert result is None
    assert not (save_dir / f"{race_id}.html").exists()


def test_get_race_ids_by_day_fetches_and_caches(tmp_path):
    html = """
    <html>
        <body>
            <a href="/race/202301010101">Race 1</a>
            <a href="/race/202301010102">Race 2</a>
            <a href="/other/link">Other Link</a>
            <a href="/race/202301010101">Duplicate Race 1</a>
        </body>
    </html>
    """

    def fetch_html(_url: str, _encoding: str) -> str | None:
        return html

    client = ScraperClient(
        base_dir=str(tmp_path), fetch_html=fetch_html, wait_fn=lambda: None
    )
    result = client.get_race_ids_by_day(date(2023, 1, 1))

    assert sorted(result) == sorted(["202301010101", "202301010102"])
    cache_path = tmp_path / "data" / "raw" / "html" / "race_lists" / "20230101.html"
    assert cache_path.read_text(encoding="utf-8") == html


def test_get_race_ids_by_day_uses_cache(tmp_path):
    cache_path = tmp_path / "data" / "raw" / "html" / "race_lists" / "20230101.html"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text('<a href="/race/202301010103">Race 3</a>', encoding="utf-8")

    def fetch_html(_url: str, _encoding: str) -> str | None:
        raise AssertionError("network should not be called")

    client = ScraperClient(
        base_dir=str(tmp_path), fetch_html=fetch_html, wait_fn=lambda: None
    )
    result = client.get_race_ids_by_day(date(2023, 1, 1))

    assert result == ["202301010103"]


def test_get_race_ids_by_day_returns_empty_on_failure(tmp_path):
    def fetch_html(_url: str, _encoding: str) -> str | None:
        return None

    client = ScraperClient(
        base_dir=str(tmp_path), fetch_html=fetch_html, wait_fn=lambda: None
    )
    result = client.get_race_ids_by_day(date(2023, 1, 1))

    assert result == []
    cache_path = tmp_path / "data" / "raw" / "html" / "race_lists" / "20230101.html"
    assert not cache_path.exists()


def test_fetch_and_save_race_html_writes_file(tmp_path):
    race_id = "123456789012"
    save_dir = tmp_path / "races"
    payload = b"Race Result HTML"

    def fetch_bytes(_url: str) -> bytes | None:
        return payload

    client = ScraperClient(fetch_bytes=fetch_bytes, wait_fn=lambda: None)
    result = client.fetch_and_save_race_html(race_id, str(save_dir))

    expected_path = save_dir / f"{race_id}.html"
    assert result == payload
    assert expected_path.read_bytes() == payload


def test_fetch_and_save_race_html_uses_cache(tmp_path):
    race_id = "123456789012"
    save_dir = tmp_path / "races"
    save_dir.mkdir(parents=True, exist_ok=True)
    expected_path = save_dir / f"{race_id}.html"
    expected_path.write_bytes(b"cached")

    def fetch_bytes(_url: str) -> bytes | None:
        raise AssertionError("network should not be called")

    client = ScraperClient(fetch_bytes=fetch_bytes, wait_fn=lambda: None)
    result = client.fetch_and_save_race_html(race_id, str(save_dir))

    assert result == b"cached"
    assert expected_path.read_bytes() == b"cached"


def test_fetch_and_save_race_html_returns_none_on_failure(tmp_path):
    race_id = "123456789012"
    save_dir = tmp_path / "races"

    def fetch_bytes(_url: str) -> bytes | None:
        return None

    client = ScraperClient(fetch_bytes=fetch_bytes, wait_fn=lambda: None)
    result = client.fetch_and_save_race_html(race_id, str(save_dir))

    assert result is None
    assert not (save_dir / f"{race_id}.html").exists()


def test_fetch_and_save_horse_html_writes_file(tmp_path):
    horse_id = "1234567890"
    save_dir = tmp_path / "horses"
    payload = b"Horse Profile HTML"

    def fetch_bytes(_url: str) -> bytes | None:
        return payload

    client = ScraperClient(fetch_bytes=fetch_bytes, wait_fn=lambda: None)
    result = client.fetch_and_save_horse_html(horse_id, str(save_dir))

    expected_path = save_dir / f"{horse_id}.html"
    assert result == payload
    assert expected_path.read_bytes() == payload


def test_fetch_and_save_horse_html_uses_cache(tmp_path):
    horse_id = "1234567890"
    save_dir = tmp_path / "horses"
    save_dir.mkdir(parents=True, exist_ok=True)
    expected_path = save_dir / f"{horse_id}.html"
    expected_path.write_bytes(b"cached")

    def fetch_bytes(_url: str) -> bytes | None:
        raise AssertionError("network should not be called")

    client = ScraperClient(fetch_bytes=fetch_bytes, wait_fn=lambda: None)
    result = client.fetch_and_save_horse_html(horse_id, str(save_dir))

    assert result == b"cached"
    assert expected_path.read_bytes() == b"cached"


def test_fetch_and_save_horse_html_returns_none_on_failure(tmp_path):
    horse_id = "1234567890"
    save_dir = tmp_path / "horses"

    def fetch_bytes(_url: str) -> bytes | None:
        return None

    client = ScraperClient(fetch_bytes=fetch_bytes, wait_fn=lambda: None)
    result = client.fetch_and_save_horse_html(horse_id, str(save_dir))

    assert result is None
    assert not (save_dir / f"{horse_id}.html").exists()


def test_fetch_and_save_jockey_html_writes_file(tmp_path):
    jockey_id = "00001"
    save_dir = tmp_path / "jockeys"
    payload = b"Jockey Profile HTML"

    def fetch_bytes(_url: str) -> bytes | None:
        return payload

    client = ScraperClient(fetch_bytes=fetch_bytes, wait_fn=lambda: None)
    result = client.fetch_and_save_jockey_html(jockey_id, str(save_dir))

    expected_path = save_dir / f"{jockey_id}.html"
    assert result == payload
    assert expected_path.read_bytes() == payload


def test_fetch_and_save_jockey_html_uses_cache(tmp_path):
    jockey_id = "00001"
    save_dir = tmp_path / "jockeys"
    save_dir.mkdir(parents=True, exist_ok=True)
    expected_path = save_dir / f"{jockey_id}.html"
    expected_path.write_bytes(b"cached")

    def fetch_bytes(_url: str) -> bytes | None:
        raise AssertionError("network should not be called")

    client = ScraperClient(fetch_bytes=fetch_bytes, wait_fn=lambda: None)
    result = client.fetch_and_save_jockey_html(jockey_id, str(save_dir))

    assert result == b"cached"
    assert expected_path.read_bytes() == b"cached"


def test_fetch_and_save_jockey_html_returns_none_on_failure(tmp_path):
    jockey_id = "00001"
    save_dir = tmp_path / "jockeys"

    def fetch_bytes(_url: str) -> bytes | None:
        return None

    client = ScraperClient(fetch_bytes=fetch_bytes, wait_fn=lambda: None)
    result = client.fetch_and_save_jockey_html(jockey_id, str(save_dir))

    assert result is None
    assert not (save_dir / f"{jockey_id}.html").exists()
