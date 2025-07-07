# ⚽ GOALdigger: Football Prophecies

> “What if football drama could be predicted?” — Welcome to GOALdigger, where AI meets the pitch!

GOALdigger is a machine learning project that predicts football match outcomes and scorelines using historical match data, team statistics, and team-specific attributes.

## 🔮 Features

- Predicts the outcome: **Home Win, Draw, or Away Win**
- Estimates **goal scorelines** for each team
- ELO-based team strength tracking
- Rolling window statistics for team form
- Dramatic match commentary generation (when big teams lose/draw)

## 🧠 Models

- **Classification**: Predicts match result (Win/Draw/Loss)
- **Regression**: Predicts expected goals for both teams

## 🚀 Getting Started

### Clone the repository

```bash
git clone https://github.com/mishraansh07/GOALdigger-Football-Prophecies.git
cd GOALdigger-Football-Prophecies


### 📂 Database Download

The full SQLite database (`database.sql`, ~200MB) is too large for GitHub.  
You can download it from Google Drive:

👉 [Click here to download the database](https://drive.google.com/uc?export=download&id=1WnXbB3Y3ON0UgLrYCXFU4svn9PDzNRzb)

> After downloading, place `database.sql` in the project root directory before running `main.py`.
