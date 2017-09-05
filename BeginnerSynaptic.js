/*
	Initial Test of Synaptic.
	Very Basic.

	Attempts to recognize the type of word entered.
*/

// Imports the Synaptic Module into the project
var synaptic = require('synaptic');

// For Network:
var inputLayer = new synaptic.Layer(5);    // Layer includes 5 Neurons
var hiddenLayer = new synaptic.Layer(4);   // Layer includes 4 Neurons
var outputLayer = new synaptic.Layer(2);   // Layer includes 2 Neurons

inputLayer.project(hiddenLayer);  // Connects inputs to the hidden layer
hiddenLayer.project(outputLayer); // Connects the hidden layer to output

// Network itself:
var network = new synaptic.Network({
	// List all layers here:
	input: inputLayer,
	hidden: [hiddenLayer],
	output: outputLayer
});

// Training the network
var learningRate = .3;
for (var i = 0; i < 20000; i++) {
	
	// Nouns (inanimate objects)
	network.activate(["paper", "nail", "stapler", "chair", "house"]);
	network.propagate(learningRate, ["noun", "inanimate"]);

	// Nouns (animate objects)
	network.activate(["boy", "girl", "cat", "dog", "tree"]);
	network.propagate(learningRate, ["noun", "animate"]);

	// Pronouns (possession)
	network.activate(["he", "she", "i", "theirs", "mine"]);
	network.propagate(learningRate, ["pronoun", "possession"])

	// Verbs (present progressive)
	network.activate(["running", "jumping", "swimming", "dancing", "hiding"]);
	network.propagate(learningRate, ["verb", "progressive"]);

	// Verbs (past)
	network.activate(["shoved", "placed", "stung", "ran", "fled"]);
	network.propagate(learningRate, ["verb", "past"]);

	// Adjectives (color)
	network.activate(["red", "orange", "yellow", "green", "blue"]);
	network.propagate(learningRate, ["adjective", "color"]);

	// Adjectives (texture)
	network.activate(["rough", "soft", "bumpy", "chalky", "scaly"]);
	network.propagate(learningRate, ["adjective", "texture"]);

	// Adverbs (degree)
	network.activate(["accidentally", "fiercely", "soon", "victoriously", "easily"]);
	network.propagate(learningRate, ["adverb", "degree"]);

	// Prepositions (place)
	network.activate(["above", "below", "behind", "down", "in"]);
	network.propagate(learningRate, ["preposition", "place"]);

	// Conjunctions (connect)
	network.activate(["and", "yet", "but", "for", "so"]);
	network.propagate(learningRate, ["conjunction", "connect"]);

	// Interjections (exclaim)
	network.activate(["ouch!", "wow!", "oops!", "hey!", "no!"]);
	network.propagate(learningRate, ["interjection", "exclaim"]);
}

// Testing the network
alert(network.activate(["standing"]));
