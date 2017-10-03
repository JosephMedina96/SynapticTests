/*
	Demo Neataptic.
	Pretty Basic.
*/

// Imports Neataptic
var neataptic = require('neataptic');

/*
	=============================================================
						CREATION OF DATA SET
	=============================================================
*/

// Data Set Categories
/*
	=============================================================
	These will be appended to the data for easier classification.
	=============================================================
*/
var noun = [1, 0, 0, 0, 0, 0, 0, 0];		 // Category #1
var pronoun = [0, 1, 0, 0, 0, 0, 0, 0];		 // Category #2
var verb = [0, 0, 1, 0, 0, 0, 0, 0];		 // Category #3
var adverb = [0, 0, 0, 1, 0, 0, 0, 0];		 // Category #4
var adjective = [0, 0, 0, 0, 1, 0, 0, 0];	 // Category #5
var conjunction = [0, 0, 0, 0, 0, 1, 0, 0];	 // Category #6
var preposition = [0, 0, 0, 0, 0, 0, 1, 0];	 // Category #7
var interjection = [0, 0, 0, 0, 0, 0, 0, 1]; // Category #8

// Data Set Items
/*
	=============================================================
	Starting with a string, each item will be classified using
	one of the above categories. All items of a certain type will
	be lumped together for ease of labelling.
	=============================================================
*/

var data = 
	"
	paper nail stapler chair house boy girl cat dog tree 
	he she his theirs mine 
	running jumping swimming dancing hiding shoved placed stung ran fled 
	accidentally fiercely soon victoriously easily 
	red orange yellow green blue rough soft bumpy chalky scaly 
	and yet but for so 
	above below behind down in 
	ouch! wow! oops! hey! no!
	";

// Splits the string into an array of words
var dataSet = data.split("").filter(function(item, i, ar){ return ar.indexOf(item) === i; });

// Onehot Encoding
/*
	==============================================================
	One-hot encoding is the act of mapping letters / words to 
	numbers.
	==============================================================
*/
var onehot = {};

for (int i = 0; i < dataSet.length; i++) {
	var zeros = Array.apply(null, Array(dataSet)).map(Number.prototype.valueOf, 0);
	zeros[i] = 1;

	var item = dataSet[i];
	onehot[item] = zeros;
}

// Categorize the data using predetermined categories
var finalDataSet = [];

var previous = dataSet[0];
for (var i = 1; i < dataSet.length; i++) {
	var next = dataSet[i]

	// Categorize and append data
	// ===========================================================
	if (i <= 10) { // Nouns

		noun.append(onehot[previous]);
		finalDataSet.push({ input: noun, output: onehot[next]});

	} else if (i > 10 && i <= 15) { // Pronouns

		pronoun.append(onehot[previous]);
		finalDataSet.push({ input: pronoun, output: onehot[next]});

	} else if (i > 15 && i <= 25) { // Verbs

		verb.append(onehot[previous]);
		finalDataSet.push({ input: verb, output: onehot[next]});

	} else if (i > 25 && i <= 30) { // Adverbs

		adverb.append(onehot[previous]);
		finalDataSet.push({ input: adverb, output: onehot[next]});

	} else if (i > 30 && i <= 40) { // Adjectives

		adjective.append(onehot[previous]);
		finalDataSet.push({ input: adjective, output: onehot[next]});

	} else if (i > 40 && i <= 45) { // Conjunctions

		conjunction.append(onehot[previous]);
		finalDataSet.push({ input: conjunction, output: onehot[next]});

	} else if (i > 45 && i <= 50) { // Prepositions

		preposition.append(onehot[previous]);
		finalDataSet.push({ input: preposition, output: onehot[next]});

	} else { // Interjections

		interjection.append(onehot[previous]);
		finalDataSet.push({ input: interjection, output: onehot[next]});

	}
	// ==========================================================
	previous = next;
}

/*
	=============================================================
						CREATION OF NETWORK
	=============================================================
*/

// For network:
var inputLayer = new neataptic.Layer.Dense(finalDataSet.length);
var hiddenLayer = new neataptic.Layer.LSTM(20);
var outputLayer = new neataptic.Layer.Dense(9);

inputLayer.connect(hiddenLayer);   // Connects inputs to the hidden layer
hiddenLayer.connect(outputLayer);  // Connects hidden layer to the output

// Network itself:
var network = new neataptic.architect.Construct([inputLayer, hiddenLayer, outputLayer]);

/*
	=============================================================
						TRAINING OF NETWORK
	=============================================================
*/

network.train(finalDataSet, {
	log: 10,
	error: 0.03,
  	iterations: 1000,
  	rate: 0.3
});

// DEBUG
alert("Data Set: " + dataSet);
alert("Final Data Set: " + finalDataSet);
