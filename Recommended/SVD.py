
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import cross_val_score
from surprise import SVD, Dataset, Reader, accuracy
from surprise.model_selection import cross_validate, train_test_split, GridSearchCV
import joblib
import pickle

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(visit_area_info):
    visit_info = visit_area_info[
        visit_area_info['VISIT_AREA_TYPE_CD'].isin(range(1, 9))
    ]
    visit_info = visit_info.groupby('VISIT_AREA_NM').filter(lambda x: len(x) > 1).reset_index(drop=True)
    visit_info['ratings'] = visit_info[['DGSTFN', 'REVISIT_INTENTION', 'RCMDTN_INTENTION']].mean(axis=1)
    visit_info['TRAVELER_ID'] = visit_info['TRAVEL_ID'].str.split('_').str[1]
    visit_info['SIDO'] = visit_info['LOTNO_ADDR'].str.split().str[0]
    
    most_frequent_visits = visit_info.groupby('LOTNO_ADDR')['VISIT_AREA_NM'].agg(lambda x: x.mode().iloc[0]).reset_index()
    visit_info = visit_info.merge(most_frequent_visits, on='LOTNO_ADDR', how='left', suffixes=('', '_most_frequent'))
    visit_info['VISIT_AREA_NM'] = visit_info['VISIT_AREA_NM_most_frequent'].fillna(visit_info['VISIT_AREA_NM'])
    visit_info.drop(columns=['VISIT_AREA_NM_most_frequent'], inplace=True)
    
    return visit_info[['TRAVELER_ID', 'VISIT_AREA_NM', 'ratings', 'SIDO']]

def save_preprocessed_data(df, file_path):
    df.to_csv(file_path, index=False)

def perform_grid_search(data):
    param_grid = {
        'n_factors': [50, 100, 200],
        'n_epochs': [10, 50],
        'lr_all': [0.01, 0.1],
        'reg_all': [0.01, 0.1],
        'reg_bu': [0.01, 0.1],
        'reg_bi': [0.01, 0.1]
    }
    grid = GridSearchCV(SVD, param_grid, measures=['RMSE', 'MAE'], cv=5, n_jobs=-1, joblib_verbose=10)
    grid.fit(data)
    return grid

def train_model(trainset):
    algo = SVD(n_factors=50, lr_all=0.01, reg_all=0.1, n_epochs=50, reg_bu=0.1, reg_bi=0.1)
    algo.fit(trainset)
    return algo

def save_model(algo, file_path):
    with open(file_path, 'wb') as file:
        pickle.dump(algo, file)

def load_model(file_path):
    return joblib.load(file_path)

def calculate_recall_at_k(user_data, k=5):
    true_positives = user_data['true_rec'].sum()
    total_interest_items = user_data['true_rec'].sum()
    recommended_items = user_data['est_rec'].head(k).sum()
    return true_positives / total_interest_items if total_interest_items > 0 else 0

def recall5_calculator(df):
    df_sorted = df.sort_values(by=['userID', 'true_rec', 'predicted_rating'], ascending=[True, False, False])
    recall_at_5_values = df_sorted.groupby('userID').apply(lambda x: calculate_recall_at_k(x, k=5))
    return recall_at_5_values.mean()

def filter_majority_sido(group):
    sido_counts = group['SIDO'].value_counts()
    if len(sido_counts) > 0:
        majority_sido = sido_counts.idxmax()
        group = group[group['SIDO'] == majority_sido]
    return group

def main():
    # Load and preprocess data
    visit_F = load_data('../1.inputdata/tn_visit_area_info_F.csv')
    df = preprocess_data(visit_F)
    
    # Save preprocessed data
    save_preprocessed_data(df, "../2.preprocessed/dfF.csv")
    
    # Prepare data for model training
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(df[['TRAVELER_ID', 'VISIT_AREA_NM', 'ratings']], reader)
    trainset, testset = train_test_split(data, test_size=0.2, random_state=42)
    
    # Perform grid search
    grid = perform_grid_search(data)
    print("Best RMSE Score:", grid.best_score['rmse'])
    print("Best Parameters:", grid.best_params['rmse'])
    
    # Train and save the model
    algo = train_model(trainset)
    save_model(algo, '../4.SaveModel/model/svd_model_F.pkl')
    
    # Load the model and make predictions
    loaded_model = load_model('../4.SaveModel/model/svd_model_F.pkl')
    predictions = loaded_model.test(testset)
    
    prediction_data = [{'userID': uid, 'itemID': iid, 'true_rating': true_r, 'predicted_rating': est} for uid, iid, true_r, est, _ in predictions]
    predictions_df = pd.DataFrame(prediction_data)
    
    # Calculate Recall@5
    recall_at_5 = recall5_calculator(predictions_df)
    print('Recall@5:', recall_at_5)
    
    # Create and save top recommendations
    top_recommendations = predictions_df[predictions_df['predicted_rating'] > predictions_df['predicted_rating'].mean()]
    top_recommendations = top_recommendations.groupby('userID').apply(filter_majority_sido).reset_index(drop=True)
    top_recommendations.to_csv('../4.SaveModel/result/final_output/F_top_recommendations.csv', index=False)

if __name__ == "__main__":
    main()

# Print current timestamp
print("Current Timestamp:", datetime.now())
