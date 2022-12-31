let [INPUT_FILE, OUTPUT_W2V, OUTPUT_TOKENS, OUTPUT_TOKEN_MODEL, OUTPUT_KUSARI] = process.argv.slice(2);

OUTPUT_W2V = OUTPUT_W2V || '../models/w2v_model.txt';
OUTPUT_TOKENS = OUTPUT_TOKENS || '../data/tknzd.txt';
OUTPUT_TOKEN_MODEL = OUTPUT_TOKEN_MODEL || '../models/tknzr.json';
OUTPUT_KUSARI = OUTPUT_KUSARI || '../models/untitled.model';

const fs = require('fs');
const Kusari = require('../index.js');
const prompt = require("prompt-sync")({ sigint: true });

let startDate = timer();
const model = Kusari.load(fs.readFileSync(__dirname + '/' + OUTPUT_KUSARI));
console.log(`Loaded model in ${startDate()}ms`);

(async () => {
	startDate = timer();
	await model.init(
		__dirname + '/' + OUTPUT_TOKEN_MODEL,
		__dirname + '/' + OUTPUT_W2V,
		__dirname + '/' + OUTPUT_TOKENS
	);
	console.log(`Initiated model in ${startDate()}ms`);
	while(true) {
		let fromPrompt = prompt("> ");
		if(fromPrompt === '') break;

		startDate = timer();
		let responses = [];
		for (let i = 0; i < 20; i++) {
			let response = await model.generate({
				from: fromPrompt,
			});
			
			responses.push(response);
		}
		console.log(`generated in ${startDate()}ms`);

		const sorted = responses.sort((a, b) => b.score - a.score);

		console.log(sorted[0]);
	}
})();

function timer(){
	let time = Date.now();
	return () => Date.now() - time;
}