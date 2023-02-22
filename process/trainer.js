const { promisify } = require('util');
const fs = require('fs');
const readline = require('readline');
const w2v = require('../node-word2vec-win32');

const { Tokenizer } = require('../tokenizers/bindings/tokenizer');
const { WordPiece } = require('../tokenizers/bindings/models');
const { whitespacePreTokenizer } = require('../tokenizers/bindings/pre-tokenizers');
const {
	sequenceNormalizer,
	lowercaseNormalizer, 
	nfdNormalizer,
	stripAccentsNormalizer
} = require('../tokenizers/bindings/normalizers');
const { templateProcessing } = require('../tokenizers/bindings/post-processors');
const { wordPieceTrainer } = require('../tokenizers/bindings/trainers');
const Kusari = require('../index.js');

const trainTokenizer = async (inputFile, outputFile) => {
	let bertTokenizer = new Tokenizer(WordPiece.init({}, { unkToken: "[UNK]" }));

	bertTokenizer.setNormalizer(sequenceNormalizer([
		nfdNormalizer(), lowercaseNormalizer(), stripAccentsNormalizer()
	]))

	bertTokenizer.setPreTokenizer(whitespacePreTokenizer());

	bertTokenizer.setPostProcessor(templateProcessing(
		"[CLS] $A [SEP]",
		"[CLS] $A [SEP] $B:1 [SEP]:1",
		[["[CLS]", 1], ["[SEP]", 2]]
	));

	let trainer = wordPieceTrainer({
		vocabSize: 30522,
		specialTokens: ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
	});

	let files = [inputFile];
	await bertTokenizer.train(files, trainer);
	await bertTokenizer.save(outputFile);

	return bertTokenizer;
};

function trainW2VModel(inputFile, outputFile) {
	return new Promise(async (resolve) => {
		w2v.word2vec(inputFile, outputFile, {
			minCount: 1,
			cbow: 0
		}, code => {
			resolve(code);
		});
	});
}

const tokenizeFile = async (tokenizer, inputFile, outputFile) => {
	const readStream = fs.createReadStream(inputFile, { encoding: "utf8" });
	await new Promise(r => readStream.once('readable', r));

	const outputStream = fs.createWriteStream(outputFile);
	const rl = readline.createInterface({
		input: readStream,
		terminal: false,
		crlfDelay: Infinity
	});

	let encode = promisify(tokenizer.encode.bind(tokenizer));

	let lineCount = 0;

	for await (const line of rl) {
		let output = await encode(line);

		outputStream.write(output.getIds().join(" ") + "\n");
		lineCount++;

		if(lineCount % 10000 === 0) process.stdout.write("\rTokenizing line #" + lineCount);
	}

	return true;
};

const trainKusari = async ({inputFile, w2vFile, tokenizerFile, tokenizedDataFile, outputFile}) => {
	const readStream = fs.createReadStream(inputFile, { encoding: "utf8" });
	readStream.on('error', console.log);

	await new Promise(r => readStream.once('readable', r));

	const model = new Kusari();
	await model.init(tokenizerFile, w2vFile, tokenizedDataFile);

	const rl = readline.createInterface({
		input: readStream,
		terminal: false
	});

	// console.log(model)

	let lineCount = 0;
	for await (const line of rl) {
		await model.update(line, {
			updateNGrams: false
		});
		lineCount++;

		if(lineCount % 10000 === 0) process.stdout.write("\rTraining on line #" + lineCount);
	}

	await fs.promises.writeFile(outputFile, model.save());

	return model;
};

const trainAll = async ({inputFile, outputW2VFile, outputTokenizedData, outputTokenizerFile, outputKusariFile}) => {
	let bertTokenizer = await trainTokenizer(inputFile, outputTokenizerFile);
	await tokenizeFile(bertTokenizer, inputFile, outputTokenizedData);
	await trainW2VModel(outputTokenizedData, outputW2VFile);
	await trainKusari({
		inputFile,
		w2vFile: outputW2VFile,
		tokenizerFile: outputTokenizerFile,
		tokenizedDataFile: outputTokenizedData,
		outputFile: outputKusariFile
	});

	return true;
};

module.exports = {
	trainAll,
		trainTokenizer,
		tokenizeFile,
		trainW2VModel,
		trainKusari
};