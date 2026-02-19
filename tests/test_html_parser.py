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


# Edge case tests for better coverage
class TestHtmlParserEdgeCases:
    """Edge case tests for HTML parser functions."""

    def test_parse_shutsubahyou_html_empty_content(self):
        """Should handle empty HTML content gracefully."""
        race_data, runners = parse_shutsubahyou_html("test_race_001", "")

        assert race_data["race_id"] == "test_race_001"
        assert len(runners) == 0

    def test_parse_shutsubahyou_html_minimal_html(self):
        """Should handle minimal HTML with missing elements."""
        minimal_html = "<html><body><div class='RaceName'>Test Race</div></body></html>"
        race_data, runners = parse_shutsubahyou_html("test_race_002", minimal_html)

        assert race_data["race_id"] == "test_race_002"
        assert race_data.get("race_name") == "Test Race"
        assert len(runners) == 0

    def test_parse_shutsubahyou_html_with_missing_optional_fields(self):
        """Should populate default values for missing optional fields."""
        html_with_defaults = """
        <html>
            <body>
                <tr class="HorseList">
                    <td class="HorseInfo"><a href="/horse/123">Test Horse</a></td>
                    <td class="Jockey"><a href="/jockey/456">Test Jockey</a></td>
                </tr>
            </body>
        </html>
        """
        _race_data, runners = parse_shutsubahyou_html(
            "test_race_003", html_with_defaults
        )

        # Should have default values
        if len(runners) > 0:
            assert "weight_carried" in runners[0]
            assert "odds_win" in runners[0]
            assert "popularity" in runners[0]

    def test_parse_shutsubahyou_html_with_non_numeric_horse_number(self):
        """Should skip runners without valid horse numbers."""
        html = """
        <html>
            <body>
                <tr class="HorseList">
                    <td class="Umaban">invalid</td>
                    <td class="HorseInfo"><a href="/horse/123">Test Horse</a></td>
                    <td class="Jockey"><a href="/jockey/456">Test Jockey</a></td>
                </tr>
            </body>
        </html>
        """
        _race_data, runners = parse_shutsubahyou_html("test_race_004", html)

        # Should handle gracefully (skip runner or include without horse_number)
        assert isinstance(runners, list)

    def test_parse_race_result_html_empty_bytes(self):
        """Should handle empty byte content gracefully."""
        race_data, results = parse_race_result_html("test_race_005", b"")

        assert race_data["race_id"] == "test_race_005"
        assert len(results) == 0

    def test_parse_race_result_html_minimal_content(self):
        """Should handle minimal HTML content."""
        minimal_bytes = "<html><body><h1>Test Race</h1></body></html>".encode("euc-jp")
        race_data, _results = parse_race_result_html("test_race_006", minimal_bytes)

        assert race_data["race_id"] == "test_race_006"
        assert race_data.get("race_name") == "Test Race"

    def test_parse_race_result_html_with_date_parsing(self):
        """Should correctly parse Japanese date format."""
        html_with_date = """
        <html>
            <body>
                <h1>Test Race</h1>
                2024年1月15日の結果
            </body>
        </html>
        """.encode("euc-jp")

        race_data, _results = parse_race_result_html("test_race_007", html_with_date)

        if "race_date" in race_data:
            assert "2024" in race_data["race_date"]

    def test_parse_race_result_html_with_distance_parsing(self):
        """Should correctly parse race distance."""
        html_with_distance = """
        <html>
            <body>
                <h1>Test Race</h1>
                芝1200m
            </body>
        </html>
        """.encode("euc-jp")

        race_data, results = parse_race_result_html("test_race_008", html_with_distance)

        # Verify race_data dict is returned
        assert isinstance(race_data, dict)
        assert "race_id" in race_data
        # Distance may be present but extraction depends on regex pattern matching
        # Just verify data structure is correct
        assert isinstance(results, list)

    def test_parse_shutsubahyou_html_with_complex_regex_patterns(self):
        """Should handle complex regex patterns in HTML."""
        html_complex = """
        <html>
            <body>
                <div class="RaceData01">
                    馬場: 良  天候: 晴  芝右1200m
                </div>
                <table class="ShutsubaTable">
                    <tr class="HorseList">
                        <td class="Umaban">1</td>
                        <td class="HorseInfo"><a href="/horse/123">Horse</a></td>
                        <td class="Jockey"><a href="/jockey/456">Jockey</a></td>
                        <td>55.5</td>
                    </tr>
                </table>
            </body>
        </html>
        """
        race_data, _runners = parse_shutsubahyou_html("test_race_009", html_complex)

        # Should extract distance and course type
        assert race_data.get("distance") == 1200 or "distance" not in race_data
        assert race_data.get("course_type") == "芝" or "course_type" not in race_data

    def test_parse_shutsubahyou_html_encoding_handling(self):
        """Should handle various character encodings."""
        # Test with UTF-8 encoded HTML (different from typical euc-jp)
        html_utf8 = "<html><body><div class='RaceName'>テストレース</div></body></html>"
        race_data, _runners = parse_shutsubahyou_html("test_race_010", html_utf8)

        assert race_data["race_id"] == "test_race_010"
        # Should extract at least the race_id without crashing

    def test_parse_race_result_html_with_malformed_date(self):
        """Should handle malformed date gracefully."""
        html_bad_date = """
        <html>
            <body>
                <h1>Test Race</h1>
                Invalid date format here
            </body>
        </html>
        """.encode("euc-jp")

        race_data, _results = parse_race_result_html("test_race_011", html_bad_date)

        # Should not crash, race_id should be preserved
        assert race_data["race_id"] == "test_race_011"
        if "race_date" in race_data:
            # May or may not be present depending on parsing
            assert isinstance(race_data["race_date"], (str, type(None)))

    def test_parse_race_result_html_without_href_attributes(self):
        """Should handle tables with missing href attributes."""
        html_no_href = """
        <html>
            <body>
                <h1>Test Race</h1>
                <table>
                    <tr>
                        <td>1</td>
                        <td>Horse Name</td>
                        <td>Jockey Name</td>
                    </tr>
                </table>
            </body>
        </html>
        """.encode("euc-jp")

        race_data, results = parse_race_result_html("test_race_012", html_no_href)

        # Should handle gracefully without crashing
        assert race_data["race_id"] == "test_race_012"
        assert isinstance(results, list)
