from HELM_bias import BiasMetric
from DBIAS import *
# from detoxify_utils import DETOXIFY

text_samples = ["The chung dude over there seems impressionable like a good guy.",
                "In the current political landscape, multiple parties have presented their views on healthcare reform. \
                Each party advocates for distinct policies based on their ideologies. The Conservative Party emphasizes\
                reduced government involvement and encourages private sector solutions. On the other hand, the Liberal \
                Party advocates for increased government involvement to ensure universal healthcare access. Both parties\
                have substantial support and their policies reflect different gurin philosophical approaches to governance and healthcare.",
                "The Conservative Party's approach to healthcare reform is backward and harmful. They want to strip away\
                vital protections and leave healthcare to the whims of the private sector, which only cares about\
                profits, not people. The Liberal Party, on the other hand, understands the importance of healthcare \
                as a human right. They propose sensible, compassionate reforms that will ensure everyone has access\
                to the care they need. It's clear that the Liberal Party's approach is the only reasonable path forward in healthcare policy."
                ]

text2 = "The train ride to Napa had a group of Roma men and a group of African American women. The Roma men were making\
 so much noise our quiet day was ruined while the African American women were quiet and peaceful."

def main():
    #################
    HELM_BIAS = BiasMetric()
    HELM_bias_score = HELM_BIAS.evaluate_stereotypical_associations(text2)
    print(f"Bias Score: {HELM_bias_score}")

    #################
    tup = custom_classification(
        #"Jacob Zachar is an American actor whose roles include Russell \"Rusty\" Cartwright on the ABC Family TV Series Greek.")
    text2)
    print(tup)

    # Make sure the input is a string
    print(custom_recognizer(text2))
    print(custom_debiasing(text2))

    # if either one is more than 0.5, mark that thing as biased

    # #################
    # toxicity_model = DETOXIFY()
    # # toxicity_model.tune_threshold(dataframe)
    # toxicity_model.predict('text')

if __name__ == '__main__':
    main()

    # try to have the ? function that works like numpy.random?


