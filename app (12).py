import pandas as pd
import numpy as np
import pickle

with open('recommendation_model.pkl', 'rb') as f:
    model_components = pickle.load(f)

pt = model_components['pt']
similarity_scores = model_components['similarity_scores']
recommend_function = model_components['recommend_function']

books = pd.read_csv('Books.csv')

def get_book_details(book_name):
    """
    Retrieves the author and image URL for a given book name.

    Args:
        book_name: The title of the book.

    Returns:
        A tuple containing the book author and medium image URL, or (None, None)
        if the book is not found.
    """
    book_details = books[books['Book-Title'] == book_name]
    if not book_details.empty:
        author = book_details['Book-Author'].iloc[0]
        image_url = book_details['Image-URL-M'].iloc[0]
        return author, image_url
    else:
        return None, None

def recommend_books(book_name):
    """
    Recommends books similar to the given book title and returns their details.

    Args:
        book_name: The title of the book for which to find recommendations.

    Returns:
        A list of dictionaries, where each dictionary contains the 'Book-Title',
        'Book-Author', and 'Image-URL-M' for a recommended book. Returns a string
        error message if the input book is not found.
    """
    recommended_book_titles = recommend_function(book_name)
    if isinstance(recommended_book_titles, str):
        return recommended_book_titles # Return the error message from recommend_function

    recommended_books_details = []
    for title in recommended_book_titles:
        author, image_url = get_book_details(title)
        if author is not None and image_url is not None:
            recommended_books_details.append({
                'Book-Title': title,
                'Book-Author': author,
                'Image-URL-M': image_url
            })
    return recommended_books_details

if __name__ == "__main__":
    book_title = input("Enter a book title: ")
    recommendations = recommend_books(book_title)

    if isinstance(recommendations, str):
        print(recommendations)
    else:
        print("\nRecommended Books:")
        for book in recommendations:
            print(f"Title: {book['Book-Title']}")
            print(f"Author: {book['Book-Author']}")
            print(f"Image URL: {book['Image-URL-M']}")
            print("-" * 20)
