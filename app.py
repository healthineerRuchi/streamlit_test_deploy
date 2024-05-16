import streamlit as st
import matplotlib.pyplot as plt


def main():
    st.title("Welcome to Streamlit!!!")

    categories = ["Category A", "Category B", "Category C", "Category D"]
    values = [23, 17, 35, 29]

    # Create a bar graph
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(categories, values, color=["blue", "green", "red", "purple"])

    # Add title and labels
    ax.set_title("Dummy Bar Graph")
    ax.set_xlabel("Categories")
    ax.set_ylabel("Values")

    # Display the plot in the Streamlit app
    st.pyplot(fig)


if __name__ == "__main__":
    main()
