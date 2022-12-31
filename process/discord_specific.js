const fs = require('fs');
const readline = require('readline');

(async () => {
	const readStream = fs.createReadStream(__dirname + '/../data/okbh_raw_logs.txt', { encoding: "utf8" });
	await new Promise(r => readStream.once('readable', r));

	const outputStream = fs.createWriteStream(__dirname + '/../data/okbh_clean_logs.txt');
	const rl = readline.createInterface({
		input: readStream,
		terminal: false,
		crlfDelay: Infinity
	});

	let lineCount = 0;

	for await (const line of rl) {
		let emojis = line.match(/:[^ :]*:/g) || [];
		let newLine = line;

		for (let i = 0; i < emojis.length; i++) {
			newLine = newLine.replace(emojis[i], emojis[i].replace(/:/g, '') + ' ᚧ');
		}

		outputStream.write(newLine + "\n");
		lineCount++;

		if(lineCount % 10000 === 0) process.stdout.write("\rCleaning line #" + lineCount);
	}
})();