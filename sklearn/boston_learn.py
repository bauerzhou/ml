import sklearn.datasets
import sklearn.linear_model
import numpy.random
import numpy.linalg
import matplotlib.pyplot as plt

if __name__ == "__main__":
	boston = sklearn.datasets.load_boston()

	sampleRatio = 0.5
	n_samples = len(boston.target)
	sampleBoundary = int(n_samples * sampleRatio)

	#shuffle the whole data
	shuffleIdx = range(n_samples)
	numpy.random.shuffle(shuffleIdx)

	#Make the training data
	train_features = boston.data[shuffleIdx[:sampleBoundary]]
	train_targets = boston.target[shuffleIdx[:sampleBoundary]]

	#Make the testing data
	test_features = boston.data[shuffleIdx[sampleBoundary:]]
	test_targets = boston.target[shuffleIdx[sampleBoundary:]]
	
	#training
	linearRegression = sklearn.linear_model.LinearRegression()
	linearRegression.fit(train_features, train_targets)

	#Predict
	predict_targets = linearRegression.predict(test_features)


	#Evaluation
	n_test_samples = len(test_targets)
	X = range(n_test_samples)

	error = numpy.linalg.norm(predict_targets - test_targets, ord = 1) / n_test_samples
	print "Ordinary least square Error: %.2f" %(error)


	plt.plot(X, predict_targets, 'r--', label = 'PredictPrice')
	plt.plot(X, test_targets, 'g:', label='TruePrice')
	legend = plt.legend()

	plt.title("Ordinary Least Squres(boston) ")
	plt.ylabel("Price")
	plt.savefig("ordinary boston.png", format='png')
	plt.show()






