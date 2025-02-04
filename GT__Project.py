from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

class RecommendationSystem:
    def __init__(self):
        self.user_items = {}
        self.graph = nx.Graph()
        self.algo = None
        self.recommendation_graph = nx.Graph()
        self.item_prices = {}
        self.user_price_limits = {}

    def add_user_item_interaction(self, user, item):
        if user not in self.user_items:
            self.user_items[user] = []
        self.user_items[user].append(item)
        self.graph.add_node(user, bipartite=0)
        self.graph.add_node(item, bipartite=1)
        self.graph.add_edge(user, item)

    def set_item_price(self, item, price):
        self.item_prices[item] = price

    def set_user_price_limit(self, user, price_limit):
        self.user_price_limits[user] = price_limit

    def train_matrix_factorization(self):
        interactions = []
        for user, items in self.user_items.items():
            for item in items:
                interactions.append((user, item, 1))

        reader = Reader(rating_scale=(1, 5))
        dataset = Dataset.load_from_df(pd.DataFrame(interactions, columns=['user', 'item', 'rating']), reader)
        trainset, _ = train_test_split(dataset, test_size=0.2)

        self.algo = SVD()
        self.algo.fit(trainset)

    def recommend_items_matrix_factorization(self, user):
        price_limit = self.user_price_limits.get(user, float('inf'))  # Fetch the user's price limit
        items_to_predict = [item for item in self.graph.nodes if item not in self.user_items[user] and item not in self.user_items]
        predictions = [(item, self.algo.predict(user, item).est) for item in items_to_predict]
        predictions.sort(key=lambda x: x[1], reverse=True)

        recommended_items = []
        total_price = 0
        for item, _ in predictions:
            if item in self.item_prices and total_price + self.item_prices[item] <= price_limit:
                recommended_items.append(item)
                total_price += self.item_prices[item]
        return recommended_items

    def visualize_graph(self):
        user_nodes = {n for n, d in self.graph.nodes(data=True) if d['bipartite'] == 0}
        item_nodes = {n for n, d in self.graph.nodes(data=True) if d['bipartite'] == 1}
        pos = nx.bipartite_layout(self.graph, user_nodes)

        nx.draw(self.graph, pos, with_labels=True, font_weight='bold')
        plt.title("User and Items Interaction Graph")
        plt.show()

    def visualize_recommendation_graph(self):
        pos = nx.bipartite_layout(self.graph, set(self.user_items.keys()))

        self.recommendation_graph.clear()
        for user in self.user_items:
            recommendations = self.recommend_items_matrix_factorization(user)
            for item in recommendations:
                self.recommendation_graph.add_node(item)
                self.recommendation_graph.add_edge(user, item)

        nx.draw(self.recommendation_graph, pos, with_labels=True, font_weight='bold')
        plt.title("User and Recommended Items Graph (With Different Price Limits)")
        plt.show()

    def recommend_items_all_users(self):
        all_recommendations = {}
        for user in self.user_items:
            recommendations = self.recommend_items_matrix_factorization(user)
            all_recommendations[user] = recommendations
        return all_recommendations

if __name__ == '__main__':
    recommendation_system = RecommendationSystem()

    while True:
        choice = input("Enter '1' to add user-item interaction\n '2' to set item price\n "
                       "'3' to set user price limit\n '4' to train the system\n "
                       "'5' to visualize the graph\n '6' to visualize recommendation graph\n "
                       "'7' to exit: \n")

        if choice == '1':
            user = int(input("Enter user ID: "))
            item = int(input("Enter item ID: "))
            recommendation_system.add_user_item_interaction(user, item)
        elif choice == '2':
            item = int(input("Enter item ID: "))
            price = float(input("Enter item price: "))
            recommendation_system.set_item_price(item, price)
        elif choice == '3':
            user = int(input("Enter user ID: "))
            price_limit = float(input("Enter price limit for the user: "))
            recommendation_system.set_user_price_limit(user, price_limit)
        elif choice == '4':
            recommendation_system.train_matrix_factorization()
            print("Matrix factorization trained.")
        elif choice == '5':
            recommendation_system.visualize_graph()
        elif choice == '6':
            recommendation_system.visualize_recommendation_graph()
        elif choice == '7':
            print("Exiting the program.")
            break
        else:
            print("Invalid choice. Please enter a valid option.")
