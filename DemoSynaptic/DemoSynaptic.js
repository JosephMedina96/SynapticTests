/*
	DemoSynaptic.js

	Uses an example provided by the creators of Synaptic
*/

var synaptic = require("synaptic");

function Perceptron(input, hidden, output) {
	// create the layers
	var inputLayer = new synaptic.Layer(input);
	var hiddenLayer = new synaptic.Layer(hidden);
	var outputLayer = new synaptic.Layer(output);

	// connect the layers
	inputLayer.project(hiddenLayer);
	hiddenLayer.project(outputLayer);

	// set the layers
	percepNetwork = new synaptic.Network({
		input: inputLayer,
		hidden: [hiddenLayer],
		output: outputLayer
	});

	return percepNetwork;
}

var myPerceptron = Perceptron(2,3,1);
var myTrainer = new synaptic.Trainer(myPerceptron);

var trainingSet = [
	{
		input: [0,0],
		output: [0]
  	},
  	{
    	input: [0,1],
    	output: [1]
  	},
  	{
    	input: [1,0],
   		output: [1]
  	},
  	{
    	input: [1,1],
    	output: [0]
  	},
];

myTrainer.train(trainingSet);

var testA = myPerceptron.activate([0,0]);
var testB = myPerceptron.activate([1,0]);
var testC = myPerceptron.activate([0,1]);
var testD = myPerceptron.activate([1,1]);

if (Math.round(testA) == 0) {
  alert("Test A is correct.");
} else {
  alert("Test A is incorrect, more training required.");
}

if (Math.round(testB) == 1) {
  alert("Test B is correct.");
} else {
  alert("Test B is incorrect, more training required.");
}

if (Math.round(testC) == 1) {
  alert("Test C is correct.");
} else {
  alert("Test C is incorrect, more training required.");
}

if (Math.round(testD) == 0) {
  alert("Test D is correct.");
} else {
  alert("Test D is incorrect, more training required.");
}
