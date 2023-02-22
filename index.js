const { Packr, pack } = require('msgpackr');
const { Tokenizer } = require('./tokenizers/bindings/tokenizer');
const { wordPieceDecoder } = require('./tokenizers/bindings/decoders');
const { promisify } = require('util');
const fs = require('fs');
const word2vec = require('./node-word2vec-win32');
const { NGram, MultipleFile, GoodTuringSmoothing } = require('./nlptoolkit-ngram');
const packageJSON = require('./package.json');

const packr = new Packr({
	structuredClone: true
});

const SIMI_CACHE = {};

class Kusari {
	constructor() {
		this.version = packageJSON.version;
		this.corpus = [];

		this.config = {
			candidateMax: 5000,
			ngrams: 3,
			seed: 'banana',
			top_k: 5,
			infrequencySimiRatio: 0.5,
			contextFrequencyLimit: 1
		};
		this.tokenFrequencies = {};
		this.tokenCount = 0;

		this.w2v = { words: 0, size: 0 };
		this.wordVectors = null;
		this.nGram = null;
		this.tokenizer = null;
		this.encodeTokens = null;
		this.decodeTokens = null;
		this.tokenRankings = {};
	}

	async init(tokenizerFile, w2vFile, tokenizedDataFile) {
		this.rand = mulberry32(cyrb128(this.config.seed).reduce((a, v) => a + v));

		this.tokenizer = await Tokenizer.fromFile(tokenizerFile);
		this.tokenizer.setDecoder(wordPieceDecoder());

		this.encodeTokens = promisify(this.tokenizer.encode.bind(this.tokenizer));
		this.decodeTokens = promisify(this.tokenizer.decode.bind(this.tokenizer));

		let simpleSmoothing = new GoodTuringSmoothing();
		this.nGram = new NGram(new MultipleFile(tokenizedDataFile).readCorpus(), this.config.ngrams);
		this.nGram.calculateNGramProbabilitiesSimple(simpleSmoothing);

		let tokenSortable = Object.keys(this.tokenFrequencies).map(t => [t, this.tokenFrequencies[t]]);
		tokenSortable = tokenSortable.sort((a, b) => b[1] - a[1]);
		for (let i = 0; i < tokenSortable.length; i++) {
			this.tokenRankings[tokenSortable[i][0]] = i;
		}

		await this.syncW2VFile(w2vFile);

		return true;
	}

	syncW2VFile(w2vFile) {
		return new Promise((resolve, reject) => {
			word2vec.loadModel(w2vFile, (error, model) => {
				if(error) return reject(error);

				this.w2v = { words: this.tokenizer.getVocabSize(), size: Number(model.size) };
				this.wordVectors = new Float32Array(this.w2v.words * this.w2v.size);

				if(this.corpus.length < this.w2v.words) {
					let padAmount = this.w2v.words - this.corpus.length;
					for (let i = 0; i < padAmount; i++) { this.corpus.push([]); }
				}

				const vectors = model.getVectors();
				for (let i = 0; i < vectors.length; i++) {
					let wordID = Number(vectors[i].word);
					this.wordVectors.set(vectors[i].values, wordID * this.w2v.size);
				}
				
				resolve(model);
			});
		});
	}

	async update(text, _settings={}) {
		const settings = {
			updateNGrams: false,
			..._settings
		};
		let tokens = await this.encodeTokens(text);
		let tokenIDs = tokens.getIds();
		for (let i = 0; i < tokenIDs.length - 1; i++) {
			if(!this.tokenFrequencies[tokenIDs[i]]) this.tokenFrequencies[tokenIDs[i]] = 0;
			this.tokenFrequencies[tokenIDs[i]]++;

			if(!this.corpus[tokenIDs[i]].includes(tokenIDs[i+1])) {
				this.corpus[tokenIDs[i]].push(tokenIDs[i+1]);
			}
		}

		this.tokenCount += tokenIDs.length;

		if(settings.updateNGrams) this.nGram.addNGramSentence(tokenIDs);

		return tokenIDs;
	}

	async generate({
		from="",
		maxlength=20
	}) {
		let currentToken;
		let fromTokensIDs;

		// If there is no from argument: Use the [CLS] starting token
		// If there is a from argument: Use the 2nd to last token in from. (Last token is a stop token)
		if(from === '') {
			currentToken = this.tokenizer.tokenToId('[CLS]');
		} else {
			let fromTokens = await this.encodeTokens(from);
			fromTokensIDs = fromTokens.getIds();
			currentToken = fromTokensIDs[fromTokensIDs.length - 2];
		}
		// Track grams so that the model doesn't repeat itself
		let oldGrams = [];
		// Track used tokens to punish word reusage
		let usedTokens = new Set();

		let responseTokens = [currentToken];
		let responseScore = 0;

		// This loop repeats until a new route cannot be found or the currentToken is invalid.
		while(currentToken) {
			// Get all the possible routes from the currentToken
			let routes = this.corpus[currentToken];
			if(routes.length === 0) break;

			// Setup the score format
			let scores = routes.map(r => [r, 0]);

			const contextTokenIDs = responseTokens.concat(fromTokensIDs);
			const rewardUnit = contextTokenIDs.length;
			for (let i = 0; i < scores.length; i++) {
				let maxScore = rewardUnit * 3;

				// Create an ngram
				let gram = responseTokens.concat([scores[i][0]]).slice(-this.config.ngrams);

				// Punishments
				if(
					(scores[i][0] === currentToken) || // Don't repeat the currentToken
					(oldGrams.includes(gram.join('_'))) || // Don't use old ngrams
					(usedTokens.has(scores[i][0])) // Don't use old tokens
				) {
					scores[i][1] -= rewardUnit;
				}

				// Calculate the probability that this ngram would occur in the dataset.
				scores[i][1] += this.nGram.getProbability(gram[0], gram[1], gram[2]) * rewardUnit;

				// Calculate the similarity to the prompt and response so far
				let infrequencyScore = (1 - this.rankedFrequency(scores[i][0])) * this.config.infrequencySimiRatio;
				for(let j = 0; j < contextTokenIDs.length; j++) {
					// Skip the check if it passes the frequency threshold (probably a stop word).
					if(this.rankedFrequency(contextTokenIDs[j]) > this.config.contextFrequencyLimit) continue;
					let simi = this.similarity(scores[i][0], contextTokenIDs[j]) * (1 - this.config.infrequencySimiRatio);
					scores[i][1] += simi + infrequencyScore;
				}

				// Score should sum to about 3 reward units by this point.
				// Normalize the score
				scores[i][1] = scores[i][1] / maxScore;
			}

			// Shuffle the array so that responses are not entirely deterministic.
			shuffleArray(scores, this.rand);
			scores = scores.sort((a, b) => b[1] - a[1]);
			scores = scores.slice(0, this.config.top_k);

			// Pick a top scoring token
			let topScore = scores[Math.floor(this.rand() * scores.length)];

			// Make it the new token if it's valid
			if(!topScore) break;
			currentToken = topScore[0];
			if(!currentToken) break;

			responseScore += topScore[1] || 0;
			responseTokens.push(topScore[0]);
			if(responseTokens.length >= maxlength) break;

			// Record gram and token
			let gram = responseTokens.slice(-this.config.ngrams);
			oldGrams.push(gram.join('_'));
			usedTokens.add(currentToken);
		}

		// Normalize response score
		responseScore /= responseTokens.length; 

		let result = await this.decodeTokens(responseTokens, true);

		return {
			result: result,
			tokens: responseTokens,
			score: responseScore
		};
	}

	routes(token, isPretokenized) {
		let tokenized = this.toTokens(token, isPretokenized)[0];
		let tokenID = this.addToDictionary(tokenized);
		return this.getCorpusArr(tokenID).map(tid => this.dictionary[tid]);
	}

	similarity(token1, token2) {
		if(token1 === token2) return 1;

		if(SIMI_CACHE[`${token1}_${token2}`]) return SIMI_CACHE[`${token1}_${token2}`];

		[token1, token2] = [token1, token2].map(Number);
		if(!this.tokenizer.idToToken(token1) || !this.tokenizer.idToToken(token2)) return 0;

		let sum = 0;
		for (let i = 0; i < this.w2v.size; i++ ) {
			sum += this.wordVectors[(token1 * this.w2v.size) + i] * this.wordVectors[(token2 * this.w2v.size) + i];
		}
		SIMI_CACHE[`${token1}_${token2}`] = sum;
		return sum;
	}

	frequency(token) {
		return this.tokenFrequencies[token] / this.tokenCount;
	}

	rankedFrequency(token) {
		if(typeof this.tokenRankings[token] === 'undefined') return 0;
		return 1 - (this.tokenRankings[token] / this.w2v.words);
	}

	save() {
		const serializable = {
			version: this.version,
			corpus: this.corpus,
			tokenFrequencies: this.tokenFrequencies,
			tokenCount: this.tokenCount,
			nGram: this.nGram
		};

		return packr.pack(serializable);
	}

	static load(buffer) {
		if(typeof buffer === 'string') buffer = fs.readFileSync(buffer);
		const data = packr.unpack(buffer);
		const model = new Kusari();

		model.version = data.version;
		model.config = {
			...model.config,
			...data.config
		};
		model.corpus = data.corpus;
		model.tokenFrequencies = data.tokenFrequencies;
		model.tokenCount = data.tokenCount;

		return model;
	}
}

function cyrb128(str) {
	let h1 = 1779033703, h2 = 3144134277,
		h3 = 1013904242, h4 = 2773480762;
	for (let i = 0, k; i < str.length; i++) {
		k = str.charCodeAt(i);
		h1 = h2 ^ Math.imul(h1 ^ k, 597399067);
		h2 = h3 ^ Math.imul(h2 ^ k, 2869860233);
		h3 = h4 ^ Math.imul(h3 ^ k, 951274213);
		h4 = h1 ^ Math.imul(h4 ^ k, 2716044179);
	}
	h1 = Math.imul(h3 ^ (h1 >>> 18), 597399067);
	h2 = Math.imul(h4 ^ (h2 >>> 22), 2869860233);
	h3 = Math.imul(h1 ^ (h3 >>> 17), 951274213);
	h4 = Math.imul(h2 ^ (h4 >>> 19), 2716044179);
	return [(h1^h2^h3^h4)>>>0, (h2^h1)>>>0, (h3^h1)>>>0, (h4^h1)>>>0];
}

function mulberry32(a) {
	return function() {
		var t = a += 0x6D2B79F5;
		t = Math.imul(t ^ t >>> 15, t | 1);
		t ^= t + Math.imul(t ^ t >>> 7, t | 61);
		return ((t ^ t >>> 14) >>> 0) / 4294967296;
	}
}

function shuffleArray(array, rand=Math.random) {
	let currentIndex = array.length, randomIndex;

	// While there remain elements to shuffle...
	while (0 !== currentIndex) {
		// Pick a remaining element...
		randomIndex = Math.floor(rand() * currentIndex);
		currentIndex--;

		// And swap it with the current element.
		[array[currentIndex], array[randomIndex]] = [
			array[randomIndex], array[currentIndex]
		];
	}

	return array;
};

module.exports = Kusari;