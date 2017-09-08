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

alert(myPerceptron.activate([0,0])); // 0.0268581547421616
alert(myPerceptron.activate([1,0])); // 0.9829673642853368
alert(myPerceptron.activate([0,1])); // 0.9831714267395621
alert(myPerceptron.activate([1,1])); // 0.02128894618097928
