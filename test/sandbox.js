let [W2V_MODEL, TOKENIZED_FILE, TOKEN_MODEL, KUSARI_MODEL] = process.argv.slice(2);

W2V_MODEL = W2V_MODEL || '../models/w2v_model.txt';
TOKENIZED_FILE = TOKENIZED_FILE || '../data/tknzd.txt';
TOKEN_MODEL = TOKEN_MODEL || '../models/tknzr.json';
KUSARI_MODEL = KUSARI_MODEL || '../models/untitled.model';

const fs = require('fs');
const Kusari = require('../index.js');
const prompt = require("prompt-sync")({ sigint: true });

let startDate = timer();
const model = Kusari.load(fs.readFileSync(__dirname + '/' + KUSARI_MODEL));
console.log(`Loaded model in ${startDate()}ms`);

(async () => {
	startDate = timer();
	await model.init(
		__dirname + '/' + TOKEN_MODEL,
		__dirname + '/' + W2V_MODEL,
		__dirname + '/' + TOKENIZED_FILE
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