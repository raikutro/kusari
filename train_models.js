let [INPUT_FILE, OUTPUT_W2V, OUTPUT_TOKENS, OUTPUT_TOKEN_MODEL, OUTPUT_KUSARI] = process.argv.slice(2);
const { trainAll, trainKusari } = require('./process/trainer');

INPUT_FILE = INPUT_FILE || 'data/clean_logs.txt';
OUTPUT_W2V = OUTPUT_W2V || 'models/w2v_model.txt';
OUTPUT_TOKENS = OUTPUT_TOKENS || 'data/tknzd.txt';
OUTPUT_TOKEN_MODEL = OUTPUT_TOKEN_MODEL || 'models/tknzr.json';
OUTPUT_KUSARI = OUTPUT_KUSARI || 'models/untitled.model';

(async () => {
	let startDate = Date.now();

	console.log('Training Kusari & all support models...');
	await trainAll({
		inputFile: __dirname + '/' + INPUT_FILE,
		outputW2VFile: __dirname + '/' + OUTPUT_W2V,
		outputTokenizedData: __dirname + '/' + OUTPUT_TOKENS,
		outputTokenizerFile: __dirname + '/' + OUTPUT_TOKEN_MODEL,
		outputKusariFile: __dirname + '/' + OUTPUT_KUSARI
	});

	console.log();
	console.log();
	console.log(`Finished creating models in ${Math.floor((Date.now() - startDate) / 1000)}s   Nice!   d-(^ ‿ ^ )ᕗ`);
})();