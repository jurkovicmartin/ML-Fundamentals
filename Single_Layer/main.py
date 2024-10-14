from single_layer import SingleLayerNetwork


def main():
    number_of_neurons = 4
    input_size = 2
    network = SingleLayerNetwork(number_of_neurons, input_size)

    input = [[0, 0],
             [0, 1],
             [1, 0],
             [1, 1]]
    # OR, AND, NOR, NAND 
    labels = [[0, 0, 1, 1],
              [1, 0, 0, 1],
              [1, 0, 0, 1],
              [1, 1, 0, 0]]
    
    network.train(input, labels, learning_rate=0.5, error=0.1, graph=True)

    input = [0, 0]
    predictions = network.predict(input)
    print(f"""Input: {input}
          OR prediction: {predictions[0]}
          AND prediction: {predictions[1]}
          NOR prediction: {predictions[2]}
          NAND prediction: {predictions[3]}""")




if __name__ == "__main__":
    main()