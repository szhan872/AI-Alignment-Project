from Bias_detection import BiasDetectionModel

# Define the main function as a demo use of single text classification
# This function takes a text input from the user and classifies it as biased or non-biased
def main():
    classifier = BiasDetectionModel()

    text = input("Enter a text: ")

    result = classifier.classify_text(text)

    print(lambda: "Biased" if result == 1 else "Non-Biased" if result == 0 else None)()
    # print("Classification result:", lambda: "Biased" if result == 0 else "Non-Biased")

if __name__ == "__main__":
    main()