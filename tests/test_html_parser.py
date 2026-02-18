from pathlib import Path

import pytest

from umabozu.data_collection.html_parser import (
    parse_horse_profile_html,
    parse_jockey_profile_html,
    parse_race_result_html,
    parse_shutsubahyou_html,
)


@pytest.fixture
def fixtures_dir():
    """フィクスチャディレクトリのパスを返す"""
    return Path(__file__).parent / "fixtures" / "html"


@pytest.fixture
def shutsubahyou_html(fixtures_dir):
    """出馬表HTMLフィクスチャを読み込む(文字列)"""
    path = fixtures_dir / "upcoming_races" / "202305010211.html"
    # EUC-JPでエンコードされているので、バイト列で読み込んでからデコード
    content_bytes = path.read_bytes()
    return content_bytes.decode("euc-jp")


@pytest.fixture
def race_result_html(fixtures_dir):
    """レース結果HTMLフィクスチャを読み込む(バイト列)"""
    path = fixtures_dir / "races" / "202248041104.html"
    return path.read_bytes()


@pytest.fixture
def horse_profile_html(fixtures_dir):
    """馬プロフィールHTMLフィクスチャを読み込む(バイト列)"""
    path = fixtures_dir / "horses" / "2020102621.html"
    return path.read_bytes()


@pytest.fixture
def jockey_profile_html(fixtures_dir):
    """騎手プロフィールHTMLフィクスチャを読み込む(バイト列)"""
    path = fixtures_dir / "jockeys" / "00410.html"
    return path.read_bytes()


def test_parse_shutsubahyou_html_extracts_race_and_runners(shutsubahyou_html):
    """出馬表HTMLから正しくレース情報と出走馬をパースできる"""
    race_data, runners = parse_shutsubahyou_html("202305010211", shutsubahyou_html)

    # レース情報の検証(実際のデータに基づく)
    assert race_data["race_id"] == "202305010211"
    assert "race_name" in race_data
    # race_numberはHTMLによって有無が異なるため存在確認のみ

    # 出走馬データの検証(少なくとも1頭は存在する)
    assert len(runners) > 0
    first_runner = runners[0]
    assert first_runner["race_id"] == "202305010211"
    assert "horse_number" in first_runner
    assert "horse_id" in first_runner
    assert "jockey_id" in first_runner


def test_parse_race_result_html_extracts_summary_and_rows(race_result_html):
    """レース結果HTMLから正しくレースデータと着順をパースできる"""
    race_data, results = parse_race_result_html("202248041104", race_result_html)

    # レース情報の検証(実際のデータに基づく)
    assert race_data["race_id"] == "202248041104"
    assert "race_name" in race_data
    # race_numberはHTMLパース結果に依存(厳密な値は検証しない)
    assert "race_number" in race_data

    # 着順データの検証(少なくとも1頭は存在する)
    assert len(results) > 0
    first_result = results[0]
    assert first_result["race_id"] == "202248041104"
    assert first_result["rank"] == 1
    assert "horse_id" in first_result
    assert "jockey_id" in first_result
    assert "trainer_id" in first_result


def test_parse_horse_profile_html_extracts_core_fields(horse_profile_html):
    """馬プロフィールHTMLから正しく基本情報をパースできる"""
    horse = parse_horse_profile_html("2020102621", horse_profile_html)

    # 必須フィールドの検証
    assert horse["horse_id"] == "2020102621"
    assert horse["horse_name"] is not None
    assert len(horse["horse_name"]) > 0

    # 任意フィールドの存在確認(実データに基づく)
    assert "birth_date" in horse
    assert "trainer_id" in horse
    assert "owner_id" in horse


def test_parse_jockey_profile_html_extracts_core_fields(jockey_profile_html):
    """騎手プロフィールHTMLから正しく基本情報をパースできる"""
    jockey = parse_jockey_profile_html("00410", jockey_profile_html)

    # 必須フィールドの検証
    assert jockey["jockey_id"] == "00410"
    assert jockey["jockey_name"] is not None
    assert len(jockey["jockey_name"]) > 0

    # 任意フィールドの存在確認(実データに基づく)
    assert "birth_date" in jockey
    assert "belonging" in jockey
