var neataptic = require('neataptic');

var $ = require('jQuery');
$.ajaxSetup({async: false});

var data = [];
var trainingSet = [];

var trainingOptions = {
	log: 10,
	error: 0.03,
	iterations: 10000,
	rate: 0.3
};

var speechParts = {};
speechParts.noun = [1,0,0,0,0,0,0,0];
speechParts.pronoun = [0,1,0,0,0,0,0,0];
speechParts.verb = [0,0,1,0,0,0,0,0];
speechParts.adverb = [0,0,0,1,0,0,0,0];
speechParts.adjective = [0,0,0,0,1,0,0,0];
speechParts.conjunction = [0,0,0,0,0,1,0,0];
speechParts.preposition = [0,0,0,0,0,0,1,0];
speechParts.interjection = [0,0,0,0,0,0,0,1];

var inputLayer = neataptic.Layer.Dense(1);
var hiddenLayer = neataptic.Layer.Dense(9);
var outputLayer = neataptic.Layer.Dense(8);

inputLayer.connect(hiddenLayer);
hiddenLayer.connect(outputLayer);

var network = new neataptic.architect.Construct([inputLayer, hiddenLayer, outputLayer]);

String.prototype.hashCode = function () {
  var hash = 0;
  if (this.length == 0) return hash;
  for (var i = 0; i < this.length; i++) {
    character = this.charCodeAt(i);
    hash = ((hash<<5)-hash)+character;
    hash = hash & hash; //Convert to 32-bit integer
  }
  return hash;
}

Number.prototype.map = function (in_min, in_max, out_min, out_max) {
  return (this - in_min) * (out_max - out_min) / (in_max - in_min) + out_min;
}

Array.prototype.compareArray = function (array) {
  if (!array) return false;
  if (this.length != array.length) return false;
  for (var i = 0; i < this.length; i++) {
    if (this[i] instanceof Array && array[i] instanceof Array) {
      if (!this[i].compare(array[i])) return false;
    } else if (this[i] != array[i]) {
      return false;
    }
  }
  return true;
}

function getFileData () {

  var fileData = new Array();

  $.get('data.txt', function(data) {
    var elements = data.split("\n");
    for (var i = 0; i < elements.length+1; i++) {
    	fileData.push(elements[i]);
    }
  });

  return fileData; 
}

data = Array.from(getFileData());

function categorize (data) {
	var current;
	for (var i = 0; i < data.length; i++) {
		current = data[i].hashCode();
		current = current.map(-1000000, 100, 0, 1);

		if (i <= 50) { // Nouns

    		trainingSet.push({ input: [current], output: speechParts.noun});

    	} else if (i > 50 && i <= 75) { // Pronouns

	    	trainingSet.push({ input: [current], output: speechParts.pronoun});

	    } else if (i > 75 && i <= 125) { // Verbs

	    	trainingSet.push({ input: [current], output: speechParts.verb});

	    } else if (i > 125 && i <= 150) { // Adverbs

	    	trainingSet.push({ input: [current], output: speechParts.adverb});

	    } else if (i > 150 && i <= 200) { // Adjectives

	    	trainingSet.push({ input: [current], output: speechParts.adjective});

	    } else if (i > 200 && i <= 225) { // Conjunctions

	    	trainingSet.push({ input: [current], output: speechParts.conjunction});

	    } else if (i > 225 && i <= 250) { // Prepositions

	    	trainingSet.push({ input: [current], output: speechParts.preposition});

	    } else { // Interjections

	    	trainingSet.push({ input: [current], output: speechParts.interjection});

	    }
	}
}

network.train(trainingSet, trainingOptions);