// Read from file
document.getElementById('file').onchange = function(){

	var file = this.files[0];
	var reader = new FileReader();
	reader.onload = function(progressEvent){
    	// Entire file
    	console.log(this.result);

    	// By lines
    	var lines = this.result.split('\n');
    	for(var line = 0; line < lines.length; line++){
      		console.log(lines[line]);
    	}
  	};
  	reader.readAsText(file);
}

// Notes
// 50 Nouns
// 25 Pronouns
// 50 Verbs
// 25 Adverbs
// 50 Adjectives
// 25 Conjunctions
// 25 Prepositions
// 10 Interjections