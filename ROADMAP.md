# ROADMAP for rank_llms

## Alternative Ranking Methods

Beyond the current ELO-based system, we could implement these alternative ranking methods:

1. **Win Rate Percentage**
   - Simple calculation of overall win percentage for each model
   - Easy to understand but doesn't account for opponent strength

2. **Weighted Win Percentage**
   - Weight wins by opponent strength
   - More wins against stronger models count more

3. **Bradley-Terry Model**
   - Statistical model for paired comparisons
   - Estimates probability of one model beating another

4. **TrueSkill™ System**
   - Similar to ELO but includes uncertainty measurements
   - Better handles incomplete comparison data

5. **Category-Weighted Rankings**
   - Weight different prompt categories differently
   - Allows customization based on what skills matter most

6. **PageRank-Style Algorithm**
   - Treat models as nodes in a graph with edges weighted by win rates
   - Models that beat strong models rank higher

7. **Tournament Points System**
   - Assign points for wins/ties (3 for win, 1 for tie)
   - Rank by total points accumulated