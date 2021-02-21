
# AI Portfolio Manager
class PortfolioManagementModel:

    def __init__(self):
        # Data in to test that the saving of weights worked
        data = pd.read_csv('IBM.csv')
        X = data['Delta Close']
        y = data.drop(['Delta Close'], axis=1)

        # Read structure from json
        json_file = open('model.json', 'r')
        json = json_file.read()
        json_file.close()
        self.network = model_from_json(json)

        # Read weights from HDF5
        self.network.load_weights("weights.h5")

        # Verify weights and structure are loaded
        y_pred = self.network.predict(X.values)
        y_pred = np.around(y_pred, 0)
        print(classification_report(y, y_pred))

PortfolioManagementModel()
