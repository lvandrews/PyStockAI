# Class to develop your AI portfolio manager
class AIPMDevelopment:

    def __init__(self):
        # Read your data in and split the dependent and independent
        data = pd.read_csv('IBM.csv')
        X = data['Delta Close']
        y = data.drop(['Delta Close'], axis=1)

        # Train test spit
        X_train, X_test, y_train, y_test = train_test_split(X, y)

        # Create the sequential
        network = Sequential()

        # Create the structure of the neural network
        network.add(Dense(1, input_shape=(1,), activation='relu'))
        network.add(Dense(3, activation='relu'))
        network.add(Dense(3, activation='relu'))
        network.add(Dense(3, activation='relu'))
        network.add(Dense(1, activation='relu'))

        # Compile the model
        network.compile(
                      optimizer='rmsprop',
                      loss='binary_crossentropy',
                      metrics=['accuracy']
                      )
        # Train the model
        network.fit(X_train.values, y_train.values, epochs=100)

        # Evaluate the predictions of the model
        y_pred = network.predict(X_test.values)
        y_pred = np.around(y_pred, 0)
        print(classification_report(y_test, y_pred))

        # Save structure to json
        model = network.to_json()
        with open("model.json", "w") as json_file:
            json_file.write(model)

        # Save weights to HDF5
        network.save_weights("weights.h5")

AIPMDevelopment()
