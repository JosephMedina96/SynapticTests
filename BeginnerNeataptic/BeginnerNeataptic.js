/*
	Initial Test of Neataptic.
	Very Basic.

	Attempts to recognize the type of word entered.
*/

// Imports the Synaptic Module into the project
var neataptic = require('neataptic');

// For Network:
var inputLayer = new neataptic.Layer.Dense(5);    // Layer includes 5 Neurons
var hiddenLayer = new neataptic.Layer.LSTM(4);   // Layer includes 4 Neurons
var outputLayer = new neataptic.Layer.Dense(2);   // Layer includes 2 Neurons

inputLayer.connect(hiddenLayer);  // Connects inputs to the hidden layer
hiddenLayer.connect(outputLayer); // Connects the hidden layer to output

// Network itself:
var network = new neataptic.architect.Construct([inputLayer, hiddenLayer, outputLayer]);

// *Data Sets*
/*
	=====================================================================
	First, we insert words relating to the different parts of speech.
	=====================================================================
*/
var inanimateNouns = "paper nail stapler chair house";
var animateNouns = "boy girl cat dog tree";
var pronouns = "he she i theirs mine";
var presentVerbs = "running jumping swimming dancing hiding";
var pastVerbs = "shoved placed stung ran fled";
var colorAdjectives = "red orange yellow green blue";
var textureAdjectives = "rough soft bumpy chalky scaly";
var adverbs = "accidentally fiercely soon victoriously easily";
var prepositions = "above below behind down in";
var conjunctions = "and yet but for so";
var interjections = "ouch! wow! oops! hey! no!";

// *Processing Data Sets*
/*
	=====================================================================
	Here, we are creating arrays of words from the data strings. To make
	identification easier, we make sure all letters are lower-cased.
	=====================================================================
*/
inanimateNouns = inanimateNouns.toLowerCase();
var inanimateNounSet = inanimateNouns.split("").filter(function(item, i, ar){ return ar.indexOf(item) === i; });

animateNouns = animateNouns.toLowerCase();
var animateNounSet = animateNouns.split("").filter(function(item, i, ar){ return ar.indexOf(item) === i; });

pronouns = pronouns.toLowerCase();
var pronounSet = pronouns.split("").filter(function(item, i, ar){ return ar.indexOf(item) === i; });

presentVerbs = presentVerbs.toLowerCase();
var presentVerbSet = presentVerbs.split("").filter(function(item, i, ar){ return ar.indexOf(item) === i; });

pastVerbs = pastVerbs.toLowerCase();
var pastVerbSet = pastVerbs.split("").filter(function(item, i, ar){ return ar.indexOf(item) === i; });

colorAdjectives = colorAdjectives.toLowerCase();
var colorAdjectiveSet = colorAdjectives.split("").filter(function(item, i, ar){ return ar.indexOf(item) === i; });

textureAdjectives = textureAdjectives.toLowerCase();
var textureAdjectiveSet = textureAdjectives.split("").filter(function(item, i, ar){ return ar.indexOf(item) === i; });

adverbs = adverbs.toLowerCase();
var adverbSet = adverbs.split("").filter(function(item, i, ar){ return ar.indexOf(item) === i; });

prepositions = prepositions.toLowerCase();
var prepositionSet = prepositions.split("").filter(function(item, i, ar){ return ar.indexOf(item) === i; });

conjunctions = conjunctions.toLowerCase();
var conjunctionSet = conjunctions.split("").filter(function(item, i, ar){ return ar.indexOf(item) === i; });

interjections = interjections.toLowerCase();
var interjectionSet = interjections.split("").filter(function(item, i, ar){ return ar.indexOf(item) === i; });

// *One-hot Encoding*
/*
	=====================================================================
	One-hot encoding is the act of mapping letters / words to numbers.
	=====================================================================
*/
var onehotINouns = {};
var onehotANouns = {};
var onehotPronouns = {};
var onehotPrVerbs = {};
var onehotPaVerbs = {};
var onehotCAdjectives = {};
var onehotTAdjectives = {};
var onehotAdverbs = {};
var onehotPrepositions = {};
var onehotConjunctions = {};
var onehotInterjections = {};

for (var i = 0; i < inanimateNounSet.length; i++) {
	var zeros = Array.apply(null, Array(inanimateNounSet)).map(Number.prototype.valueOf, 0);
	zeros[i] = 1;

	var iNoun = inanimateNouns[i];
	onehotINouns[iNoun] = zeros;
}

for (var i = 0; i < animateNounSet.length; i++) {
	var zeros = Array.apply(null, Array(animateNounSet)).map(Number.prototype.valueOf, 0);
	zeros[i] = 1;

	var aNoun = animateNouns[i];
	onehotANouns[aNoun] = zeros;
}

for (var i = 0; i < pronounSet.length; i++) {
	var zeros = Array.apply(null, Array(pronounSet)).map(Number.prototype.valueOf, 0);
	zeros[i] = 1;

	var proNoun = pronouns[i];
	onehotPronouns[proNoun] = zeros;
}

for (var i = 0; i < presentVerbSet.length; i++) {
	var zeros = Array.apply(null, Array(presentVerbSet)).map(Number.prototype.valueOf, 0);
	zeros[i] = 1;

	var prVerb = presentVerbs[i];
	onehotPrVerbs[prVerb] = zeros;
}

for (var i = 0; i < pastVerbSet.length; i++) {
	var zeros = Array.apply(null, Array(pastVerbSet)).map(Number.prototype.valueOf, 0);
	zeros[i] = 1;

	var paVerb = pastVerbs[i];
	onehotPaVerbs[paVerb] = zeros;
}

for (var i = 0; i < colorAdjectiveSet.length; i++) {
	var zeros = Array.apply(null, Array(colorAdjectiveSet)).map(Number.prototype.valueOf, 0);
	zeros[i] = 1;

	var cAdj = colorAdjectives[i];
	onehotCAdjectives[cAdj] = zeros;
}

for (var i = 0; i < textureAdjectiveSet.length; i++) {
	var zeros = Array.apply(null, Array(textureAdjectiveSet)).map(Number.prototype.valueOf, 0);
	zeros[i] = 1;

	var tAdj = textureAdjectives[i];
	onehotTAdjectives[tAdj] = zeros;
}

for (var i = 0; i < adverbSet.length; i++) {
	var zeros = Array.apply(null, Array(adverbSet)).map(Number.prototype.valueOf, 0);
	zeros[i] = 1;

	var adVerb = adverbs[i];
	onehotAdverbs[adVerb] = zeros;
}

for (var i = 0; i < prepositionSet.length; i++) {
	var zeros = Array.apply(null, Array(prepositionSet)).map(Number.prototype.valueOf, 0);
	zeros[i] = 1;

	var prPos = prepositions[i];
	onehotPrepositions[prPos] = zeros;
}

for (var i = 0; i < conjunctionSet.length; i++) {
	var zeros = Array.apply(null, Array(conjunctionSet)).map(Number.prototype.valueOf, 0);
	zeros[i] = 1;

	var cnJun = conjunctions[i];
	onehotConjunctions[cnJun] = zeros;
}

for (var i = 0; i < interjectionSet.length; i++) {
	var zeros = Array.apply(null, Array(interjectionSet)).map(Number.prototype.valueOf, 0);
	zeros[i] = 1;

	var inTer = interjections[i];
	onehotInterjections[inTer] = zeros;
}

// *Preparing Main Training Set*
/*
	=====================================================================
	Now, we need to unify the data sets into one training set for the 
	neural network.
	=====================================================================
*/
var trainingSet = [];

var previousINoun = inanimateNouns[0];
var previousANoun = animateNouns[0];
var previousPronoun = pronouns[0];
var previousPrVerb = presentVerbs[0];
var previousPaVerb = pastVerbs[0];
var previousCAdjective = colorAdjectives[0];
var previousTAdjective = textureAdjectives[0];
var previousAdverb = adverbs[0];
var previousPreposition = prepositions[0];
var previousConjunction = conjunctions[0];
var previousInterjection = interjections[0];

for (var i = 1; i < inanimateNouns.length; i++) {
	var next = inanimateNouns[i];

	trainingSet.push({ input: onehotINouns[previousINoun], output: onehotINouns[next]});
	previousINoun = next;
}

for (var i = 1; i < animateNouns.length; i++) {
	var next = animateNouns[i];

	trainingSet.push({ input: onehotANouns[previousANoun], output: onehotANouns[next]});
	previousANoun = next;
}

for (var i = 1; i < pronouns.length; i++) {
	var next = pronouns[i];

	trainingSet.push({ input: onehotPronouns[previousPronoun], output: onehotPronouns[next]});
	previousPronoun = next;
}

for (var i = 1; i < presentVerbs.length; i++) {
	var next = presentVerbs[i];

	trainingSet.push({ input: onehotPrVerbs[previousPrVerb], output: onehotPrVerbs[next]});
	previousPrVerb = next;
}

for (var i = 1; i < pastVerbs.length; i++) {
	var next = pastVerbs[i];

	trainingSet.push({ input: onehotPaVerbs[previousPaVerb], output: onehotPaVerbs[next]});
	previousPaVerb = next;
}

for (var i = 1; i < colorAdjectives.length; i++) {
	var next = colorAdjectives[i];

	trainingSet.push({ input: onehotCAdjectives[previousCAdjective], output: onehotCAdjectives[next]});
	previousCAdjective = next;
}

for (var i = 1; i < textureAdjectives.length; i++) {
	var next = textureAdjectives[i];

	trainingSet.push({ input: onehotTAdjectives[previousTAdjective], output: onehotTAdjectives[next]});
	previousTAdjective = next;
}

for (var i = 1; i < adverbs.length; i++) {
	var next = adverbs[i];

	trainingSet.push({ input: onehotAdverbs[previousAdverb], output: onehotAdverbs[next]});
	previousAdverb = next;
}

for (var i = 1; i < prepositions.length; i++) {
	var next = prepositions[i];

	trainingSet.push({ input: onehotPrepositions[previousPreposition], output: onehotPrepositions[next]});
	previousPreposition = next;
}

for (var i = 1; i < conjunctions.length; i++) {
	var next = conjunctions[i];

	trainingSet.push({ input: onehotConjunctions[previousConjunction], output: onehotConjunctions[next]});
	previousConjunction = next;
}

for (var i = 1; i < interjections.length; i++) {
	var next = interjections[i];

	trainingSet.push({ input: onehotInterjections[previousInterjection], output: onehotInterjections[next]});
	previousInterjection = next;
}

// Training the network //
/*
	=====================================================================
	Finally, we need to train the network to recognize the information
	in the training set. Hopefully it will learn the patterns based on
	the limited data we've given it.
	=====================================================================
*/
var learningRate = .3;
network.train(trainingSet, {
	clear: true
});

// Testing the network
alert(network.activate(["standing"]));
