import sqlite3 as sq3
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, log_loss
from sklearn.utils import class_weight
import numpy as np
import joblib
from tqdm import tqdm
from datetime import datetime, timedelta
import random

from team_aliases import team_name_aliases

DATABASE_PATH = "database.sqlite"
ROLLING_WINDOW_SIZE = 5
ELO_K_FACTOR = 30
INITIAL_ELO = 1500
CLASSIFIER_EPOCHS = 300
REGRESSOR_EPOCHS = 300
LEARNING_RATE = 0.01
DROPOUT_RATE = 0.2
TRAIN_TEST_SPLIT_RATIO = 0.8

def main():
    with sq3.connect(DATABASE_PATH) as connect:
        matches = pd.read_sql_query("""
        SELECT
            match_api_id,
            home_team_api_id,
            away_team_api_id,
            home_team_goal,
            away_team_goal,
            season,
            date
        FROM Match
        WHERE home_team_goal IS NOT NULL AND away_team_goal IS NOT NULL
        ORDER BY date ASC;
        """, connect)

        matches['date'] = pd.to_datetime(matches['date'])

        matches['result'] = matches.apply(
            lambda row: 0 if row['home_team_goal'] > row['away_team_goal']
            else 1 if row['home_team_goal'] == row['away_team_goal']
            else 2, axis=1
        )

        team_attrs = pd.read_sql_query("SELECT * FROM Team_Attributes", connect)
        team_attrs['date'] = pd.to_datetime(team_attrs['date'])

        team_names_df = pd.read_sql_query("SELECT team_api_id, team_long_name FROM Team", connect)

    selected_static_features = [
        'buildUpPlaySpeed', 'buildUpPlayPassing',
        'chanceCreationPassing', 'chanceCreationShooting',
        'defencePressure', 'defenceAggression'
    ]

    def create_match_rolling_features(matches_df, window_size=ROLLING_WINDOW_SIZE):
        matches_with_features = matches_df.copy()
        matches_with_features = matches_with_features.sort_values('date').reset_index(drop=True)
        all_match_features = []
        team_stats_history = {}

        for idx, match in tqdm(matches_with_features.iterrows(), total=len(matches_with_features), desc="Creating Rolling Features"):
            home_team = match['home_team_api_id']
            away_team = match['away_team_api_id']

            if home_team not in team_stats_history:
                team_stats_history[home_team] = {'goals_scored': [], 'goals_conceded': [], 'wins': [], 'draws': [], 'losses': []}
            if away_team not in team_stats_history:
                team_stats_history[away_team] = {'goals_scored': [], 'goals_conceded': [], 'wins': [], 'draws': [], 'losses': []}

            home_gs_avg = np.mean(team_stats_history[home_team]['goals_scored'][-window_size:]) if team_stats_history[home_team]['goals_scored'] else 0
            home_gc_avg = np.mean(team_stats_history[home_team]['goals_conceded'][-window_size:]) if team_stats_history[home_team]['goals_conceded'] else 0
            home_win_rate = np.mean(team_stats_history[home_team]['wins'][-window_size:]) if team_stats_history[home_team]['wins'] else 0.33
            home_draw_rate = np.mean(team_stats_history[home_team]['draws'][-window_size:]) if team_stats_history[home_team]['draws'] else 0.33
            home_loss_rate = np.mean(team_stats_history[home_team]['losses'][-window_size:]) if team_stats_history[home_team]['losses'] else 0.33

            away_gs_avg = np.mean(team_stats_history[away_team]['goals_scored'][-window_size:]) if team_stats_history[away_team]['goals_scored'] else 0
            away_gc_avg = np.mean(team_stats_history[away_team]['goals_conceded'][-window_size:]) if team_stats_history[away_team]['goals_conceded'] else 0
            away_win_rate = np.mean(team_stats_history[away_team]['wins'][-window_size:]) if team_stats_history[away_team]['wins'] else 0.33
            away_draw_rate = np.mean(team_stats_history[away_team]['draws'][-window_size:]) if team_stats_history[away_team]['draws'] else 0.33
            away_loss_rate = np.mean(team_stats_history[away_team]['losses'][-window_size:]) if team_stats_history[away_team]['losses'] else 0.33

            current_match_features = match.to_dict()
            current_match_features.update({
                'home_avg_goals_scored': home_gs_avg,
                'home_avg_goals_conceded': home_gc_avg,
                'home_win_rate': home_win_rate,
                'home_draw_rate': home_draw_rate,
                'home_loss_rate': home_loss_rate,
                'away_avg_goals_scored': away_gs_avg,
                'away_avg_goals_conceded': away_gc_avg,
                'away_win_rate': away_win_rate,
                'away_draw_rate': away_draw_rate,
                'away_loss_rate': away_loss_rate,
            })
            all_match_features.append(current_match_features)

            team_stats_history[home_team]['goals_scored'].append(match['home_team_goal'])
            team_stats_history[home_team]['goals_conceded'].append(match['away_team_goal'])
            team_stats_history[home_team]['wins'].append(1 if match['result'] == 0 else 0)
            team_stats_history[home_team]['draws'].append(1 if match['result'] == 1 else 0)
            team_stats_history[home_team]['losses'].append(1 if match['result'] == 2 else 0)

            team_stats_history[away_team]['goals_scored'].append(match['away_team_goal'])
            team_stats_history[away_team]['goals_conceded'].append(match['home_team_goal'])
            team_stats_history[away_team]['wins'].append(1 if match['result'] == 2 else 0)
            team_stats_history[away_team]['draws'].append(1 if match['result'] == 1 else 0)
            team_stats_history[away_team]['losses'].append(1 if match['result'] == 0 else 0)

        return pd.DataFrame(all_match_features)


    def calculate_elo_ratings(matches_df, k=ELO_K_FACTOR, initial_elo=INITIAL_ELO):
        elo_ratings = {team_id: initial_elo for team_id in pd.concat([matches_df['home_team_api_id'], matches_df['away_team_api_id']]).unique()}
        elo_history = []

        for index, match in tqdm(matches_df.iterrows(), total=len(matches_df), desc="Calculating Elo Ratings"):
            home_team_id = match['home_team_api_id']
            away_team_id = match['away_team_api_id']
            home_goals = match['home_team_goal']
            away_goals = match['away_team_goal']

            R_home = elo_ratings.get(home_team_id, initial_elo)
            R_away = elo_ratings.get(away_team_id, initial_elo)

            E_home = 1 / (1 + 10 ** ((R_away - R_home) / 400))
            E_away = 1 / (1 + 10 ** ((R_home - R_away) / 400))

            if home_goals > away_goals:
                S_home, S_away = 1, 0
            elif home_goals == away_goals:
                S_home, S_away = 0.5, 0.5
            else:
                S_home, S_away = 0, 1

            elo_history.append({
                'match_api_id': match['match_api_id'],
                'date': match['date'],
                'home_team_api_id': home_team_id,
                'away_team_api_id': away_team_id,
                'home_elo_before': R_home,
                'away_elo_before': R_away
            })

            new_R_home = R_home + k * (S_home - E_home)
            new_R_away = R_away + k * (S_away - E_away)

            elo_ratings[home_team_id] = new_R_home
            elo_ratings[away_team_id] = new_R_away

        return pd.DataFrame(elo_history)

    matches_with_rolling_features = create_match_rolling_features(matches)

    elo_df = calculate_elo_ratings(matches)

    final_matches = pd.merge(matches_with_rolling_features, elo_df[['match_api_id', 'home_elo_before', 'away_elo_before']],
                                on='match_api_id', how='left')
    final_matches.rename(columns={'home_elo_before': 'home_team_elo',
                                     'away_elo_before': 'away_team_elo'}, inplace=True)
    final_matches['elo_difference'] = final_matches['home_team_elo'] - final_matches['away_team_elo']

    team_attrs_sorted = team_attrs.sort_values('date')

    temp_attrs_home = team_attrs_sorted[['team_api_id', 'date'] + selected_static_features].copy()
    temp_attrs_home.rename(columns={
        'team_api_id': 'merge_team_api_id_home',
        'date': 'attr_date_home'
    }, inplace=True)
    temp_attrs_home.rename(columns={col: f'home_{col}' for col in selected_static_features}, inplace=True)

    final_matches = pd.merge_asof(
        final_matches,
        temp_attrs_home,
        left_on='date',
        right_on='attr_date_home',
        left_by='home_team_api_id',
        right_by='merge_team_api_id_home',
        direction='backward'
    )
    final_matches.drop(columns=['attr_date_home', 'merge_team_api_id_home'], inplace=True, errors='ignore')

    temp_attrs_away = team_attrs_sorted[['team_api_id', 'date'] + selected_static_features].copy()
    temp_attrs_away.rename(columns={
        'team_api_id': 'merge_team_api_id_away',
        'date': 'attr_date_away'
    }, inplace=True)
    temp_attrs_away.rename(columns={col: f'away_{col}' for col in selected_static_features}, inplace=True)

    final_matches = pd.merge_asof(
        final_matches,
        temp_attrs_away,
        left_on='date',
        right_on='attr_date_away',
        left_by='away_team_api_id',
        right_by='merge_team_api_id_away',
        direction='backward'
    )
    final_matches.drop(columns=['attr_date_away', 'merge_team_api_id_away'], inplace=True, errors='ignore')

    if 'date_x' in final_matches.columns:
        final_matches.rename(columns={'date_x': 'date'}, inplace=True)
    if 'date_y' in final_matches.columns:
        final_matches.drop(columns=['date_y'], inplace=True, errors='ignore')

    initial_rows = len(final_matches)
    final_matches.dropna(inplace=True)

    feature_columns = [f'home_{col}' for col in selected_static_features] + \
                      [f'away_{col}' for col in selected_static_features]

    rolling_feature_cols = [
        'home_avg_goals_scored', 'home_avg_goals_conceded', 'home_win_rate', 'home_draw_rate', 'home_loss_rate',
        'away_avg_goals_scored', 'away_avg_goals_conceded', 'away_win_rate', 'away_draw_rate', 'away_loss_rate'
    ]
    feature_columns.extend(rolling_feature_cols)

    feature_columns.extend(['home_team_elo', 'away_team_elo', 'elo_difference'])

    missing_cols = [col for col in feature_columns if col not in final_matches.columns]
    if missing_cols:
        raise ValueError(f"Missing feature columns after merge: {missing_cols}")

    final_matches = final_matches.sort_values('date').reset_index(drop=True)

    split_idx = int(len(final_matches) * TRAIN_TEST_SPLIT_RATIO)

    X_train_df = final_matches.iloc[:split_idx][feature_columns]
    X_test_df_original = final_matches.iloc[split_idx:].copy()
    X_test_df = X_test_df_original[feature_columns]

    y_train_classification = final_matches.iloc[:split_idx]['result']
    y_test_classification = final_matches.iloc[split_idx:]['result']

    y_train_regression = final_matches.iloc[:split_idx][['home_team_goal', 'away_team_goal']]
    y_test_regression = final_matches.iloc[split_idx:][['home_team_goal', 'away_team_goal']]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_df)
    X_test_scaled = scaler.transform(X_test_df)

    joblib.dump(scaler, "scaler_new_features.pkl")

    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_train_classification_tensor = torch.tensor(y_train_classification.values, dtype=torch.long)
    y_test_classification_tensor = torch.tensor(y_test_classification.values, dtype=torch.long)
    y_train_regression_tensor = torch.tensor(y_train_regression.values, dtype=torch.float32)
    y_test_regression_tensor = torch.tensor(y_test_regression.values, dtype=torch.float32)

    num_features = X_train_tensor.shape[1]
    num_classes = 3

    classification_model = nn.Sequential(
        nn.Linear(num_features, 128),
        nn.ReLU(),
        nn.Dropout(DROPOUT_RATE),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Dropout(DROPOUT_RATE),
        nn.Linear(64, num_classes)
    )

    class_weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train_classification_tensor.numpy()),
        y=y_train_classification_tensor.numpy()
    )
    weights_tensor = torch.tensor(class_weights, dtype=torch.float32)
    classification_loss_fn = nn.CrossEntropyLoss(weight=weights_tensor)
    classification_optimizer = optim.Adam(classification_model.parameters(), lr=LEARNING_RATE)

    for epoch in range(CLASSIFIER_EPOCHS):
        classification_model.train()
        preds = classification_model(X_train_tensor)
        loss = classification_loss_fn(preds, y_train_classification_tensor)

        classification_optimizer.zero_grad()
        loss.backward()
        classification_optimizer.step()

        if epoch % 50 == 0 or epoch == CLASSIFIER_EPOCHS -1:
            print(f"[Classifier] Epoch {epoch}/{CLASSIFIER_EPOCHS}, Loss: {loss.item():.4f}")

    torch.save(classification_model.state_dict(), "result_model_improved.pt")

    classification_model.eval()
    with torch.no_grad():
        test_preds_logits = classification_model(X_test_tensor)
        predicted_classes = torch.argmax(test_preds_logits, dim=1)

        accuracy = accuracy_score(y_test_classification.values, predicted_classes.numpy())
        print(f"\n🔍 Classification Accuracy: {accuracy * 100:.2f}%")

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test_classification.values, predicted_classes.numpy(), average='weighted', zero_division=0
        )
        print(f"📊 Weighted Precision: {precision:.2f}")
        print(f"📊 Weighted Recall: {recall:.2f}")
        print(f"📊 Weighted F1-Score: {f1:.2f}")

        cm = confusion_matrix(y_test_classification.values, predicted_classes.numpy())
        print("\nConfusion Matrix:")
        print(cm)
        label_map_eval = {0: "Home Win", 1: "Draw", 2: "Away Win"}
        print(f"Rows: Actual | Columns: Predicted")
        print(f"             {label_map_eval[0]:<10} {label_map_eval[1]:<10} {label_map_eval[2]:<10}")
        for i, row in enumerate(cm):
            print(f"{label_map_eval[i]:<10} {row[0]:<10} {row[1]:<10} {row[2]:<10}")

        test_preds_probs = torch.softmax(test_preds_logits, dim=1).numpy()
        logloss = log_loss(y_test_classification.values, test_preds_probs)
        print(f"📈 Log Loss: {logloss:.4f}")


    goal_model = nn.Sequential(
        nn.Linear(num_features, 128),
        nn.ReLU(),
        nn.Dropout(DROPOUT_RATE),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Dropout(DROPOUT_RATE),
        nn.Linear(64, 2)
    )

    goal_loss_fn = nn.MSELoss()
    goal_optimizer = optim.Adam(goal_model.parameters(), lr=LEARNING_RATE)

    for epoch in range(REGRESSOR_EPOCHS):
        goal_model.train()
        preds = goal_model(X_train_tensor)
        loss = goal_loss_fn(preds, y_train_regression_tensor)

        goal_optimizer.zero_grad()
        loss.backward()
        goal_optimizer.step()

        if epoch % 50 == 0 or epoch == REGRESSOR_EPOCHS -1:
            print(f"[GoalRegressor] Epoch {epoch}/{REGRESSOR_EPOCHS}, Loss: {loss.item():.4f}")

    torch.save(goal_model.state_dict(), "goal_model_improved.pt")

    goal_model.eval()
    with torch.no_grad():
        test_goal_preds = goal_model(X_test_tensor)

        predicted_home_goals_eval = torch.round(test_goal_preds[:, 0]).int()
        predicted_away_goals_eval = torch.round(test_goal_preds[:, 1]).int()

        actual_home_goals = y_test_regression_tensor[:, 0].int()
        actual_away_goals = y_test_regression_tensor[:, 1].int()

        mae_home = torch.mean(torch.abs(predicted_home_goals_eval.float() - actual_home_goals.float())).item()
        mae_away = torch.mean(torch.abs(predicted_away_goals_eval.float() - actual_away_goals.float())).item()
        print(f"\n⚽ Goal Prediction MAE (Home): {mae_home:.2f}")
        print(f"⚽ Goal Prediction MAE (Away): {mae_away:.2f}")

        exact_score_accuracy = ((predicted_home_goals_eval == actual_home_goals) & (predicted_away_goals_eval == actual_away_goals)).float().mean().item()
        print(f"🎯 Exact Scoreline Accuracy: {exact_score_accuracy * 100:.2f}%")

        print("\nSample Goal Predictions (Test Set):")
        for i in range(10):
            test_row = X_test_df_original.iloc[i]
            
            home_team_id = test_row['home_team_api_id']
            away_team_id = test_row['away_team_api_id']

            home_team_name = team_names_df[team_names_df['team_api_id'] == home_team_id]['team_long_name'].values[0]
            away_team_name = team_names_df[team_names_df['team_api_id'] == away_team_id]['team_long_name'].values[0]

            print(f"Predicted: {predicted_home_goals_eval[i].item()} - {predicted_away_goals_eval[i].item()} | "
                  f"Actual: {home_team_name} {actual_home_goals[i].item()} - {actual_away_goals[i].item()} {away_team_name}")


    def get_team_id_from_name(team_input_name, team_df, aliases):
        standardized_name = team_input_name.lower()
        
        if standardized_name in aliases:
            db_name = aliases[standardized_name]
            match = team_df[team_df['team_long_name'] == db_name]
            if not match.empty:
                return match['team_api_id'].values[0]

        match = team_df[team_df['team_long_name'].str.lower() == standardized_name]
        if not match.empty:
            return match['team_api_id'].values[0]
            
        return None

    scaler_loaded = joblib.load("scaler_new_features.pkl")

    classification_model_loaded = nn.Sequential(
        nn.Linear(num_features, 128), nn.ReLU(), nn.Dropout(DROPOUT_RATE),
        nn.Linear(128, 64), nn.ReLU(), nn.Dropout(DROPOUT_RATE),
        nn.Linear(64, num_classes)
    )
    classification_model_loaded.load_state_dict(torch.load("result_model_improved.pt"))
    classification_model_loaded.eval()

    goal_model_loaded = nn.Sequential(
        nn.Linear(num_features, 128), nn.ReLU(), nn.Dropout(DROPOUT_RATE),
        nn.Linear(128, 64), nn.ReLU(), nn.Dropout(DROPOUT_RATE),
        nn.Linear(64, 2)
    )
    goal_model_loaded.load_state_dict(torch.load("goal_model_improved.pt"))
    goal_model_loaded.eval()


    def get_team_attributes_for_prediction(team_id, current_date):
        latest_attrs = team_attrs[
            (team_attrs['team_api_id'] == team_id) & (team_attrs['date'] <= current_date)
        ].sort_values('date', ascending=False)
        
        if not latest_attrs.empty:
            return latest_attrs[selected_static_features].iloc[0]
        return None

    def get_rolling_features_for_prediction(team_id, historical_matches_df, window_size=ROLLING_WINDOW_SIZE):
        team_history = historical_matches_df[
            (historical_matches_df['home_team_api_id'] == team_id) |
            (historical_matches_df['away_team_api_id'] == team_id)
        ].sort_values('date', ascending=False).head(window_size)

        if team_history.empty:
            return {'avg_goals_scored': 0, 'avg_goals_conceded': 0,
                    'win_rate': 0.33, 'draw_rate': 0.33, 'loss_rate': 0.33}

        goals_scored = []
        goals_conceded = []
        wins = []
        draws = []
        losses = []

        for _, row in team_history.iterrows():
            if row['home_team_api_id'] == team_id:
                goals_scored.append(row['home_team_goal'])
                goals_conceded.append(row['away_team_goal'])
                if row['result'] == 0: wins.append(1)
                elif row['result'] == 1: draws.append(1)
                else: losses.append(1)
            else:
                goals_scored.append(row['away_team_goal'])
                goals_conceded.append(row['home_team_goal'])
                if row['result'] == 2: wins.append(1)
                elif row['result'] == 1: draws.append(1)
                else: losses.append(1)

        return {
            'avg_goals_scored': np.mean(goals_scored) if goals_scored else 0,
            'avg_goals_conceded': np.mean(goals_conceded) if goals_conceded else 0,
            'win_rate': np.mean(wins) if wins else 0.33,
            'draw_rate': np.mean(draws) if draws else 0.33,
            'loss_rate': np.mean(losses) if losses else 0.33
        }

    def get_elo_for_prediction(team_id, elo_df_history, prediction_date, initial_elo=INITIAL_ELO):
        team_elo_history_filtered = elo_df_history[
            ((elo_df_history['home_team_api_id'] == team_id) | (elo_df_history['away_team_api_id'] == team_id)) &
            (elo_df_history['date'] < prediction_date)
        ].sort_values('date', ascending=False)

        if team_elo_history_filtered.empty:
            return initial_elo
        else:
            last_elo_entry = team_elo_history_filtered.iloc[0]
            if last_elo_entry['home_team_api_id'] == team_id:
                return last_elo_entry['home_elo_before']
            else:
                return last_elo_entry['away_elo_before']

    home_name_input = input("🏟️ Enter Home Team Name: ").strip()
    away_name_input = input("🚗 Enter Away Team Name: ").strip()
    
    date_str = input("📅 Enter Prediction Date (YYYY-MM-DD): ").strip()
    try:
        prediction_date = datetime.strptime(date_str, '%Y-%m-%d')
    except ValueError:
        print("Invalid date format. Please use YYYY-MM-DD.")
        return

    home_id = get_team_id_from_name(home_name_input, team_names_df, team_name_aliases)
    away_id = get_team_id_from_name(away_name_input, team_names_df, team_name_aliases)

    if home_id is None or away_id is None:
        print("Invalid team name entered. Please check spelling or if the team exists in the database/aliases.")
        return

    home_static_attrs = get_team_attributes_for_prediction(home_id, prediction_date)
    away_static_attrs = get_team_attributes_for_prediction(away_id, prediction_date)

    if home_static_attrs is None or away_static_attrs is None:
        print("⚠️ Team attributes not found for one or both specified teams for the given date. Cannot predict.")
        print("Please ensure the Team_Attributes table has entries for these teams before or on the prediction date.")
        return

    historical_matches_for_rolling = matches[matches['date'] < prediction_date]

    home_rolling_feats = get_rolling_features_for_prediction(home_id, historical_matches_for_rolling, ROLLING_WINDOW_SIZE)
    away_rolling_feats = get_rolling_features_for_prediction(away_id, historical_matches_for_rolling, ROLLING_WINDOW_SIZE)

    home_elo_current = get_elo_for_prediction(home_id, elo_df, prediction_date)
    away_elo_current = get_elo_for_prediction(away_id, elo_df, prediction_date)
    elo_diff_current = home_elo_current - away_elo_current

    prediction_data = {
        **{f'home_{col}': home_static_attrs[col] for col in selected_static_features},
        **{f'away_{col}': away_static_attrs[col] for col in selected_static_features},
        'home_avg_goals_scored': home_rolling_feats['avg_goals_scored'],
        'home_avg_goals_conceded': home_rolling_feats['avg_goals_conceded'],
        'home_win_rate': home_rolling_feats['win_rate'],
        'home_draw_rate': home_rolling_feats['draw_rate'],
        'home_loss_rate': home_rolling_feats['loss_rate'],
        'away_avg_goals_scored': away_rolling_feats['avg_goals_scored'],
        'away_avg_goals_conceded': away_rolling_feats['avg_goals_conceded'],
        'away_win_rate': away_rolling_feats['win_rate'],
        'away_draw_rate': away_rolling_feats['draw_rate'],
        'away_loss_rate': away_rolling_feats['loss_rate'],
        'home_team_elo': home_elo_current,
        'away_team_elo': away_elo_current,
        'elo_difference': elo_diff_current
    }

    prediction_row = pd.DataFrame([prediction_data], columns=feature_columns)

    X_pred_scaled = scaler_loaded.transform(prediction_row)
    X_pred_tensor = torch.tensor(X_pred_scaled, dtype=torch.float32)

    label_map = {0: "Home Win", 1: "Draw", 2: "Away Win"}
    classification_model_loaded.eval()
    goal_model_loaded.eval()

    with torch.no_grad():
        classification_output = classification_model_loaded(X_pred_tensor)
        predicted_class_idx = torch.argmax(classification_output, dim=1).item()
        predicted_result_classifier = label_map[predicted_class_idx]

        prediction_probabilities = torch.softmax(classification_output, dim=1).squeeze().numpy()
        home_win_prob = prediction_probabilities[0] * 100
        draw_prob = prediction_probabilities[1] * 100
        away_win_prob = prediction_probabilities[2] * 100

        goal_output = goal_model_loaded(X_pred_tensor)
        predicted_home_goals = int(round(goal_output[0][0].item()))
        predicted_away_goals = int(round(goal_output[0][1].item()))

        predicted_result_consistent = predicted_result_classifier

        if predicted_home_goals == predicted_away_goals:
            predicted_result_consistent = "Draw"
        elif predicted_home_goals > predicted_away_goals:
            predicted_result_consistent = "Home Win"
        else:
            predicted_result_consistent = "Away Win"

    print(f"\n--- Prediction for {home_name_input} vs {away_name_input} ---")
    print(f"Predicted Match Result (Consistent with Score): {predicted_result_consistent}")
    print(f"Original Classifier Probabilities: Home Win: {home_win_prob:.2f}%, Draw: {draw_prob:.2f}%, Away Win: {away_win_prob:.2f}%")
    print(f"Predicted Scoreline: {home_name_input} {predicted_home_goals} - {predicted_away_goals} {away_name_input}")

    drama_teams = [
        "Manchester United", "Arsenal", "Chelsea", "Liverpool", "Tottenham Hotspur", "Manchester City",
        "Real Madrid", "Barcelona", "FC Bayern München", "Juventus", "Internazionale", "Milan", "Paris Saint-Germain",
        "Borussia Dortmund", "Atlético Madrid"
    ]

    home_team_db_name = team_name_aliases.get(home_name_input.lower(), home_name_input)
    away_team_db_name = team_name_aliases.get(away_name_input.lower(), away_name_input)

    trigger_drama = False
    drama_type = ""

    if predicted_result_consistent == "Draw" and \
       (predicted_home_goals <= 1 and predicted_away_goals <= 1) and \
       (home_team_db_name in drama_teams or away_team_db_name in drama_teams):
        trigger_drama = True
        drama_type = "draw"

    elif (home_team_db_name in drama_teams and predicted_result_consistent == "Away Win") or \
         (away_team_db_name in drama_teams and predicted_result_consistent == "Home Win"):
        trigger_drama = True
        drama_type = "loss"


    if trigger_drama:
        print("\n\n--- 🎭 BORING THINGS DESERVE DRAMA! 🎭 ---")
        if drama_type == "draw":
            print(f"Hold the presses! A {predicted_home_goals}-{predicted_away_goals} draw involving "
                  f"{home_name_input} and {away_name_input}? ")
            print("The tension! The anticipation! The sheer... lack of goals!")
            print("This isn't just a match; it's a profound philosophical statement on the nature of excitement. "
                  "Or maybe, just maybe, it was a perfectly executed defensive masterclass. Who are we to judge?")
            print("The fans will be talking about this one for minutes. Possibly even hours.")
        elif drama_type == "loss":
            losing_team_name = ""
            winning_team_name = ""
            if predicted_result_consistent == "Away Win":
                losing_team_name = home_name_input
                winning_team_name = away_name_input
            else:
                losing_team_name = away_name_input
                winning_team_name = home_name_input

            print(f"Oh, the humanity! {losing_team_name} has fallen! A shocking {predicted_home_goals}-{predicted_away_goals} result against {winning_team_name}.")
            print("The titans tremble, the foundations shake! Is this a tactical blunder, a moment of poetic injustice, or simply... a bad day at the office?")
            print("The whispers will turn into shouts, the headlines will write themselves. The drama, oh the drama, of a major team's unexpected defeat!")
        print("---------------------------------------------")

        jokes = [
            "Why don't scientists trust atoms? Because they make up everything!",
            "What do you call a fake noodle? An impasta!",
            "Why did the scarecrow win an award? Because he was outstanding in his field!",
            "I told my wife she was drawing her eyebrows too high. She looked surprised.",
            "What do you call a snowman with a six-pack? An abdominal snowman!"
        ]

        selected_joke = random.choice(jokes)
        print(f"\nHere's a random joke for you: {selected_joke}") # Print the joke instead of playing audio


    print("\nScript finished.")

if __name__ == '__main__':
    main()
