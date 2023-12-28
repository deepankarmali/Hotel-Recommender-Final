from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from geopy.distance import geodesic

df = pd.read_csv('HOTEL DATA.csv')

df.columns = df.columns.str.strip()
df['HOTELNAME'] = df['HOTELNAME'].str.strip()

hotel_names = df['HOTELNAME'].tolist()

df['features'] = df['CITY'] + ' ' + df['COUNTRY'] + ' ' + df['PROPERTYTYPE'] + ' ' + df['STARRATING'].astype(str)

tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(df['features'])

cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)

app = Flask(__name__)

def get_recommendations(hotel_name, num_recommendations=3, cosine_similarities=cosine_similarities, df=df):
    hotel_indices = df[df['HOTELNAME'].str.lower().str.strip() == hotel_name.lower().strip()].index

    if not hotel_indices.empty:
        hotel_index = hotel_indices[0]

        distances = df.apply(lambda row: geodesic((df.loc[hotel_index, 'LATITUDE'], df.loc[hotel_index, 'LONGITUDE']),
                                                  (row['LATITUDE'], row['LONGITUDE'])).km, axis=1)

        normalized_distances = 1 - (distances / distances.max())

        combined_similarity = cosine_similarities[hotel_index] + normalized_distances.values

        top_recommendations = sorted(enumerate(combined_similarity), key=lambda x: x[1], reverse=True)[:num_recommendations]

        recommended_hotels = [{
            'HOTELNAME': df.loc[hotel[0], 'HOTELNAME'],
            'CITY': df.loc[hotel[0], 'CITY'],
            'COUNTRY': df.loc[hotel[0], 'COUNTRY'],
            'PROPERTYTYPE': df.loc[hotel[0], 'PROPERTYTYPE'],
            'STARRATING': df.loc[hotel[0], 'STARRATING'],
            'RATE': df.loc[hotel[0], 'RATE'],
        } for hotel in top_recommendations]


        return recommended_hotels
    else:
        return []


@app.route('/')
def index():
    return render_template('index.html', hotel_names=hotel_names)

@app.route('/recommend', methods=['POST'])
def recommend():
    if request.method == 'POST':
        hotel_name = request.form['hotel_name']
        num_recommendations = 5
        recommendations = get_recommendations(hotel_name, num_recommendations)
        return render_template('recommendations.html', hotel_name=hotel_name, recommendations=recommendations)


if __name__ == '__main__':
    app.run(debug=True)

