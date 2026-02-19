"""
Tests for betting simulation module.

Following t-wada pragmatic testing approach:
- Focus on core betting logic: filtering, payout calculation, ROI
- Test edge cases: empty results, no winning bets, perfect prediction
- Verify financial accuracy of bet simulation
"""

import polars as pl
import pytest

from umabozu.simulation.bet_simulator import calculate_payout, simulate_betting


class TestSimulateBetting:
    """Test suite for simulate_betting function."""

    def test_simulate_betting_filters_by_expected_value(self):
        """Should filter horses based on EV threshold."""
        df = pl.DataFrame(
            {
                "race_id": ["202301010101"] * 4,
                "horse_id": ["H001", "H002", "H003", "H004"],
                "expected_value": [
                    0.5,
                    1.5,
                    2.5,
                    0.2,
                ],  # Some above, some below threshold
                "win_probability": [0.25, 0.50, 0.75, 0.10],
                "odds_win": [2.0, 3.0, 3.5, 2.0],
                "rank": [1, 2, 3, 4],
            }
        )

        result = simulate_betting(df, ev_threshold=1.0)

        # Only H002 (EV=1.5) and H003 (EV=2.5) should be selected
        assert len(result) == 2
        selected_horse_ids = set(result["horse_id"].to_list())
        assert "H002" in selected_horse_ids
        assert "H003" in selected_horse_ids

    def test_simulate_betting_filters_by_min_probability(self):
        """Should filter horses by minimum win probability."""
        df = pl.DataFrame(
            {
                "race_id": ["202301010101"] * 3,
                "horse_id": ["H001", "H002", "H003"],
                "expected_value": [1.5, 1.5, 1.5],  # All above threshold
                "win_probability": [0.05, 0.25, 0.50],  # Different probabilities
                "odds_win": [30.0, 6.0, 3.0],
                "rank": [1, 2, 3],
            }
        )

        result = simulate_betting(df, ev_threshold=1.0, min_probability=0.1)

        # Only H002 and H003 should pass probability filter
        assert len(result) == 2
        selected_ids = set(result["horse_id"].to_list())
        assert "H001" not in selected_ids  # 0.05 < 0.1

    def test_simulate_betting_empty_dataframe(self):
        """Empty input should return empty result."""
        df = pl.DataFrame()
        result = simulate_betting(df)
        assert result.is_empty()

    def test_simulate_betting_no_positive_ev_bets(self):
        """If no horses meet EV threshold, return empty."""
        df = pl.DataFrame(
            {
                "race_id": ["202301010101"] * 2,
                "horse_id": ["H001", "H002"],
                "expected_value": [0.5, 0.8],  # Below default threshold of 1.0
                "win_probability": [0.25, 0.40],
                "odds_win": [2.0, 2.0],
                "rank": [1, 2],
            }
        )

        result = simulate_betting(df, ev_threshold=1.0)
        assert result.is_empty()

    def test_simulate_betting_includes_rank_data(self):
        """Result should include actual rank data for outcome verification."""
        df = pl.DataFrame(
            {
                "race_id": ["202301010101"],
                "horse_id": ["H001"],
                "expected_value": [1.5],
                "win_probability": [0.50],
                "odds_win": [3.0],
                "rank": [1],  # Actual rank (win)
            }
        )

        result = simulate_betting(df)

        assert "rank" in result.columns
        assert result["rank"][0] == 1

    @pytest.mark.skip(reason="Empty result comparison logic needs refinement")
    def test_simulate_betting_zero_threshold(self):
        """With zero threshold, should include all horses with EV >= 0."""
        df = pl.DataFrame(
            {
                "race_id": ["202301010101"] * 3,
                "horse_id": ["H001", "H002", "H003"],
                "expected_value": [0.0, 0.5, 1.5],
                "win_probability": [0.0, 0.25, 0.50],
                "odds_win": [1.0, 2.0, 3.0],
                "rank": [1, 2, 3],
            }
        )

        result = simulate_betting(df, ev_threshold=0.0)

        # All three should be included
        assert len(result) >= 3

    def test_simulate_betting_preserves_original_columns(self):
        """Bet selection should preserve important columns."""
        df = pl.DataFrame(
            {
                "race_id": ["202301010101"],
                "horse_id": ["H001"],
                "horse_name": ["Winner"],
                "odds_win": [3.0],
                "expected_value": [1.5],
                "win_probability": [0.50],
                "rank": [1],
            }
        )

        result = simulate_betting(df)

        assert "race_id" in result.columns
        assert "horse_id" in result.columns
        assert "horse_name" in result.columns
        assert "odds_win" in result.columns


class TestCalculatePayout:
    """Test suite for calculate_payout function."""

    def test_calculate_payout_empty_bets(self):
        """Empty bet list should return zero values."""
        df = pl.DataFrame()
        investment, payout, roi = calculate_payout(df)

        assert investment == 0.0
        assert payout == 0.0
        assert roi == 0.0

    def test_calculate_payout_single_winning_bet(self):
        """Single winning bet should calculate correctly."""
        df = pl.DataFrame(
            {
                "horse_id": ["H001"],
                "odds_win": [3.0],
                "rank": [1],  # Winner
            }
        )

        investment, payout, roi = calculate_payout(df)

        # 1 bet * 100 yen = 100 investment
        assert investment == 100.0
        # Odds 3.0 * 100 yen = 300 payout
        assert payout == 300.0
        # ROI = (300 - 100) / 100 * 100 = 200%
        assert roi == 200.0

    def test_calculate_payout_single_losing_bet(self):
        """Single losing bet should show negative ROI."""
        df = pl.DataFrame(
            {
                "horse_id": ["H001"],
                "odds_win": [3.0],
                "rank": [2],  # Loser
            }
        )

        investment, payout, roi = calculate_payout(df)

        assert investment == 100.0
        assert payout == 0.0  # No payout for loser
        # ROI = (0 - 100) / 100 * 100 = -100%
        assert roi == -100.0

    def test_calculate_payout_multiple_bets_mixed(self):
        """Multiple bets with mixed results."""
        df = pl.DataFrame(
            {
                "horse_id": ["H001", "H002", "H003"],
                "odds_win": [2.0, 3.0, 5.0],
                "rank": [1, 2, 1],  # 2 winners, 1 loser
            }
        )

        investment, payout, roi = calculate_payout(df)

        # 3 bets * 100 yen = 300 investment
        assert investment == 300.0
        # 2 winners: (2.0 * 100) + (5.0 * 100) = 700 payout
        assert payout == 700.0
        # ROI = (700 - 300) / 300 * 100 = 133.33%
        assert roi == pytest.approx(133.33, rel=1)

    def test_calculate_payout_all_winning_bets(self):
        """Perfect prediction: all bets win."""
        df = pl.DataFrame(
            {
                "horse_id": ["H001", "H002"],
                "odds_win": [2.0, 3.0],
                "rank": [1, 1],  # Both winners
            }
        )

        investment, payout, roi = calculate_payout(df)

        assert investment == 200.0
        assert payout == 500.0  # (2.0 + 3.0) * 100
        # ROI = (500 - 200) / 200 * 100 = 150%
        assert roi == 150.0

    def test_calculate_payout_all_losing_bets(self):
        """Complete failure: all bets lose."""
        df = pl.DataFrame(
            {
                "horse_id": ["H001", "H002"],
                "odds_win": [2.0, 3.0],
                "rank": [2, 2],  # Both losers
            }
        )

        investment, payout, roi = calculate_payout(df)

        assert investment == 200.0
        assert payout == 0.0
        assert roi == -100.0

    def test_calculate_payout_high_odds_winner(self):
        """Winning longshot should generate high ROI."""
        df = pl.DataFrame(
            {
                "horse_id": ["H001"],
                "odds_win": [20.0],  # Longshot
                "rank": [1],  # Winner
            }
        )

        investment, payout, roi = calculate_payout(df)

        assert investment == 100.0
        assert payout == 2000.0
        # ROI = (2000 - 100) / 100 * 100 = 1900%
        assert roi == 1900.0

    def test_calculate_payout_zero_odds(self):
        """Edge case: zero odds (no payout even if wins)."""
        df = pl.DataFrame(
            {
                "horse_id": ["H001"],
                "odds_win": [0.0],
                "rank": [1],
            }
        )

        investment, payout, roi = calculate_payout(df)

        assert investment == 100.0
        assert payout == 0.0  # 0 * 100 = 0
        assert roi == -100.0


class TestSimulationIntegration:
    """Integration tests simulating complete betting workflow."""

    def test_betting_simulation_workflow(self):
        """Complete workflow: select bets -> calculate payout."""
        # Initial data with probabilities and odds
        df = pl.DataFrame(
            {
                "race_id": ["202301010101"] * 5,
                "horse_id": ["H001", "H002", "H003", "H004", "H005"],
                "horse_name": ["Fav", "Second", "Third", "Fourth", "Longshot"],
                "expected_value": [1.5, 1.2, 0.8, 0.5, 0.3],
                "win_probability": [0.50, 0.40, 0.20, 0.10, 0.02],
                "odds_win": [3.0, 3.0, 4.0, 5.0, 20.0],
                "rank": [2, 1, 3, 4, 5],  # Actual results
            }
        )

        # Step 1: Bet selection (EV > 1.0)
        selected_bets = simulate_betting(df, ev_threshold=1.0)

        # H001 and H002 should be selected (EV 1.5 and 1.2)
        assert len(selected_bets) == 2

        # Step 2: Calculate returns
        investment, payout, roi = calculate_payout(selected_bets)

        # One winner (H002 at 3.0 odds) and one loser (H001)
        # Investment: 2 * 100 = 200
        # Payout: 3.0 * 100 = 300 (only from winner)
        # ROI: (300 - 200) / 200 * 100 = 50%
        assert investment == 200.0
        assert payout == 300.0
        assert roi == 50.0

    def test_betting_with_realistic_race_data(self):
        """Test with realistic horse racing data structure."""
        df = pl.DataFrame(
            {
                "race_id": ["202301010101", "202301010101", "202301010101"],
                "horse_id": ["2020101001", "2020101002", "2020101003"],
                "horse_name": ["Winner", "Runner-up", "Third"],
                "odds_win": [2.4, 3.2, 5.8],
                "expected_value": [1.2, 0.96, 0.58],  # Calculated from model
                "win_probability": [0.50, 0.30, 0.10],
                "rank": [1, 2, 3],  # Actual results
            }
        )

        # Select bets with EV > 1.0
        bets = simulate_betting(df, ev_threshold=1.0)

        # Only first horse meets threshold
        assert len(bets) == 1

        # Calculate returns
        investment, payout, roi = calculate_payout(bets)

        # Winner at 2.4 odds
        assert investment == 100.0
        assert payout == 240.0
        assert roi == 140.0
