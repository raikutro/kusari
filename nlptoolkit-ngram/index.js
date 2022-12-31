module.exports = {
	AdditiveSmoothing: require("nlptoolkit-ngram/dist/AdditiveSmoothing").AdditiveSmoothing,
	GoodTuringSmoothing: require("nlptoolkit-ngram/dist/GoodTuringSmoothing").GoodTuringSmoothing,
	InterpolatedSmoothing: require("nlptoolkit-ngram/dist/InterpolatedSmoothing").InterpolatedSmoothing,
	LaplaceSmoothing: require("nlptoolkit-ngram/dist/LaplaceSmoothing").LaplaceSmoothing,
	MultipleFile: require("nlptoolkit-ngram/dist/MultipleFile").MultipleFile,
	NGram: require("nlptoolkit-ngram/dist/NGram").NGram,
	NGramNode: require("nlptoolkit-ngram/dist/NGramNode").NGramNode,
	NoSmoothing: require("nlptoolkit-ngram/dist/NoSmoothing").NoSmoothing,
	NoSmoothingWithDictionary: require("nlptoolkit-ngram/dist/NoSmoothingWithDictionary").NoSmoothingWithDictionary,
	NoSmoothingWithNonRareWords: require("nlptoolkit-ngram/dist/NoSmoothingWithNonRareWords").NoSmoothingWithNonRareWords,
	SimpleSmoothing: require("nlptoolkit-ngram/dist/SimpleSmoothing").SimpleSmoothing,
	TrainedSmoothing: require("nlptoolkit-ngram/dist/TrainedSmoothing").TrainedSmoothing,
};