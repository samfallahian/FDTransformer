import pickle

# Open the pickle file in binary mode
with open('_data_train_autoencoder.pickle', 'rb') as f:
    # Create a pickle Unpickler object
    unpickler = pickle.Unpickler(f)

    try:
        # Iterate over the objects in the pickle file
        while True:
            # Load the next object
            obj = unpickler.load()

            # Process the object or gather information
            # Print the type of the object as an example
            print(type(obj))
    except EOFError:
        # End of file reached
        pass

