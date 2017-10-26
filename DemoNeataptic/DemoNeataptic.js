/*
  Demo Neataptic.
  Pretty Basic.
*/

// Imports Neataptic
var neataptic = require('neataptic');

// Imports JQuery
var $ = require("jQuery");

// For Later:
// String.prototype.hashCode: Returns a number from a string
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

// Number.prototype.map: Maps numbers from one range to a new range
Number.prototype.map = function (in_min, in_max, out_min, out_max) {
  return (this - in_min) * (out_max - out_min) / (in_max - in_min) + out_min;
}

// Array.prototype.compareArray: Compares two arrays
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

/*
  =============================================================
            CREATION OF DATA SET
  =============================================================
*/

// Data Set Categories
/*
  =============================================================
  These will be outputs to the data for easier classification.
  =============================================================
*/
var noun = [1, 0, 0, 0, 0, 0, 0, 0];     // Category #1
var pronoun = [0, 1, 0, 0, 0, 0, 0, 0];    // Category #2
var verb = [0, 0, 1, 0, 0, 0, 0, 0];     // Category #3
var adverb = [0, 0, 0, 1, 0, 0, 0, 0];     // Category #4
var adjective = [0, 0, 0, 0, 1, 0, 0, 0];  // Category #5
var conjunction = [0, 0, 0, 0, 0, 1, 0, 0];  // Category #6
var preposition = [0, 0, 0, 0, 0, 0, 1, 0];  // Category #7
var interjection = [0, 0, 0, 0, 0, 0, 0, 1]; // Category #8
var blank = [];                           // Blank Category

// Data Set Items
/*
  =============================================================
  Starting with a string, each item will be classified using
  one of the above categories. All items of a certain type will
  be lumped together for ease of labelling.
  =============================================================
*/

// var data = "paper nail stapler chair house boy girl cat dog tree he she his theirs mine running jumping swimming dancing hiding shoved placed stung ran fled accidentally fiercely soon victoriously easily red orange yellow green blue rough soft bumpy chalky scaly and yet but for so above below behind down in ouch! wow! oops! hey! no!";

// Splits the string into an array of words
var dataSet = [];

// Reads a text file containing the data
// Reads a text file containing the data
document.getElementById('file').onchange = function(){

  var file = this.files[0];
  var reader = new FileReader();
  reader.onload = function(progressEvent){
    $.get(file, function(data) {
      dataSet = data.split("\n");
    });
  };
  reader.readAsText(file);

  // DEBUG
  alert("Data Set: " + dataSet);
}

// Categorize the data using predetermined categories
var finalDataSet = [];

var previous = dataSet[0];
for (var i = 1; i < dataSet.length; i++) {
  var blankData = dataSet[i-1].hashCode()/10000000000;
  var next = dataSet[i];

  if (blankData < 0) {blankData = blankData * -1}
  blank.push(blankData); // Add data for classification

  // DEBUG
  // alert("Blank: " + blank);

  // Categorize data
  // ===========================================================
  if (i <= 50) { // Nouns

    finalDataSet.push({ input: [blank], output: [1,0,0,0,0,0,0,0]});

  } else if (i > 50 && i <= 75) { // Pronouns

    finalDataSet.push({ input: [blank], output: [0,1,0,0,0,0,0,0]});

  } else if (i > 75 && i <= 125) { // Verbs

    finalDataSet.push({ input: [blank], output: [0,0,1,0,0,0,0,0]});

  } else if (i > 125 && i <= 150) { // Adverbs

    finalDataSet.push({ input: [blank], output: [0,0,0,1,0,0,0,0]});

  } else if (i > 150 && i <= 200) { // Adjectives

    finalDataSet.push({ input: [blank], output: [0,0,0,0,1,0,0,0]});

  } else if (i > 200 && i <= 225) { // Conjunctions

    finalDataSet.push({ input: [blank], output: [0,0,0,0,0,1,0,0]});

  } else if (i > 225 && i <= 250) { // Prepositions

    finalDataSet.push({ input: [blank], output: [0,0,0,0,0,0,1,0]});

  } else { // Interjections

    finalDataSet.push({ input: [blank], output: [0,0,0,0,0,0,0,1]});

  }
  // ==========================================================
  blank.pop(); // Prepares data slot for next piece of data
  previous = next;
}

// DEBUG
// alert("Final Data Set Size: " + finalDataSet.length);

/*
  =============================================================
            CREATION OF NETWORK
  =============================================================
*/

// For network:
var inputLayer = new neataptic.Layer.Dense(1);
var hiddenLayer = new neataptic.Layer.Dense(9);
var outputLayer = new neataptic.Layer.Dense(8);

inputLayer.connect(hiddenLayer);   // Connects inputs to the hidden layer
hiddenLayer.connect(outputLayer);  // Connects hidden layer to the output

// Network itself:
var network = new neataptic.architect.Construct([inputLayer, hiddenLayer, outputLayer]);

// Tells the script whether the network has been trained
var isNotTrained = true;

/*
  =============================================================
            TESTING OF NETWORK
  =============================================================
*/

// Assigns each part of the pre-results to 1 or 0
function categorizeResult (preResults) {

  var results = [];
  var highest;

  // Map the results between 0 and 1
  for (var i = 0; i < preResults.length; i++) {
    var set = preResults[i];
    var item = set.map(-9.5, 10, 0, 1);

    results.push(item);

    // DEBUG
    // alert(item);
  }

  // Find the highest value
  for (var i = 1; i < results.length; i++) {
    var previous = results[i-1];
    var current = results[i];
    
    if (current > previous) {

      highest = current;

    } else if (previous >= current) {

      highest = previous;

    } else {
      // Nothing
    }
  }

  // Set only the highest value to 1, and all others to 0
  for (var i = 0; i < results.length; i++) {
    if (results[i] == highest) {
      results[i] = 1;
    } else {
      results[i] = 0;
    }
  }

  // DEBUG
  // alert(results);

  return results;
}

// Displays the category that the system thinks the word fits in
function displayResult (test, results) {

  if (results.compareArray([1,0,0,0,0,0,0,0])) {

    alert("The system thinks that " + test + " is a noun.");
    document.getElementById('result').innerHTML = "The system thinks that " + test + " is a noun.";
    document.getElementById('result').style.color = "red";

  } else if (results.compareArray([0,1,0,0,0,0,0,0])) {

    alert("The system thinks that " + test + " is a pronoun.");
    document.getElementById('result').innerHTML = "The system thinks that " + test + " is a pronoun.";
    document.getElementById('result').style.color = "orange";

  } else if (results.compareArray([0,0,1,0,0,0,0,0])) {

    alert("The system thinks that " + test + " is a verb.");
    document.getElementById('result').innerHTML = "The system thinks that " + test + " is a verb.";
    document.getElementById('result').style.color = "yellow";

  } else if (results.compareArray([0,0,0,1,0,0,0,0])) {

    alert("The system thinks that " + test + " is an adverb.");
    document.getElementById('result').innerHTML = "The system thinks that " + test + " is an adverb.";
    document.getElementById('result').style.color = "green";

  } else if (results.compareArray([0,0,0,0,1,0,0,0])) {

    alert("The system thinks that " + test + " is an adjective.");
    document.getElementById('result').innerHTML = "The system thinks that " + test + " is an adjective.";
    document.getElementById('result').style.color = "blue";

  } else if (results.compareArray([0,0,0,0,0,1,0,0])) {

    alert("The system thinks that " + test + " is a conjunction.");
    document.getElementById('result').innerHTML = "The system thinks that " + test + " is a conjunction.";
    document.getElementById('result').style.color = "indigo";

  } else if (results.compareArray([0,0,0,0,0,0,1,0])) {

    alert("The system thinks that " + test + " is a preposition.");
    document.getElementById('result').innerHTML = "The system thinks that " + test + " is a preposition.";
    document.getElementById('result').style.color = "violet";

  } else if (results.compareArray([0,0,0,0,0,0,0,1])) {

    alert("The system thinks that " + test + " is an interjection.");
    document.getElementById('result').innerHTML = "The system thinks that " + test + " is an interjection.";
    document.getElementById('result').style.color = "brown";

  } else {

    alert("The system has failed to identify this word's category.");
    document.getElementById('result').innerHTML = "The system has failed to identify this word's category.";
    document.getElementById('result').style.color = "black";

  }
}

document.getElementById('result').innerHTML = "[Result]";
document.body.style.color = "black";

// For Future Tests:
testForPart = function () {

  if (isNotTrained) {
    /*
    =============================================================
              TRAINING OF NETWORK
    =============================================================
    */

    network.train(finalDataSet, {
      log: 10,          // Logs the activity of the network every x iterations
      error: 0.03,      // The desired error state
      iterations: 10000, // Runs the training data through the network x times
      rate: 0.3         // The speed of training
    });

    isNotTrained = false;
  }

  var data = document.getElementById('value').value;
  var preResult = network.activate([data.hashCode()]);
  var result = categorizeResult(preResult);
  displayResult(data, result);
}

userTrain = function () {
  var nounBox = document.getElementById('noun').checked;
  var pronounBox = document.getElementById('pronoun').checked;
  var verbBox = document.getElementById('verb').checked;
  var adverbBox = document.getElementById('adverb').checked;
  var adjectiveBox = document.getElementById('adjective').checked;
  var conjunctionBox = document.getElementById('conjunction').checked;
  var prepositionBox = document.getElementById('preposition').checked;
  var interjectionBox = document.getElementById('interjection').checked;

  // DEBUG
  // alert(nounBox);

  var input = document.getElementById('data').value;
  var modifiedInput = input.hashCode()/10000000000;
  var output = [
    nounBox, 
    pronounBox, 
    verbBox, 
    adverbBox, 
    adjectiveBox, 
    conjunctionBox,
    prepositionBox,
    interjectionBox
  ];

  for (var i = 0; i < output.length; i++) {
    if (output[i] == true) {
      output[i] = 1;
    } else {
      output[i] = 0;
    }
  }

  // DEBUG
  // alert(modifiedInput);
  // alert(output);

  var trainingData = { input: [modifiedInput], output: output };
  finalDataSet.push(trainingData);

  alert(input + " has been added to the training set.");
}

// ================================================================
