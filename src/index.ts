import { kmeans } from 'ml-kmeans';
import { Embedding, genkit } from 'genkit';
import { vertexAI, gemini15Pro, textEmbedding004 } from '@genkit-ai/vertexai';
import { readFileSync, writeFileSync } from 'fs';

const CATEGROIES_COUNT = 8;
const MAPPED_RESULT_FILE = 'mapped.json';

const ai = genkit({
  plugins: [
    vertexAI({ location: 'us-central1' }),

  ],
  model: gemini15Pro,
})

const generateEmbedding = async (text: string): Promise<Embedding> => {
  const data = await ai.embed({
    embedder: textEmbedding004,
    content: text,
    options: {
      taskType: "CLUSTERING",
    }
  });
  return data;
}

const mapClusters = (clusters: number[], text: string[]): Map<number, string[]> => {
  const map = new Map<number, string[]>();

  clusters.map((el, i) => {
    let value: string[] = [];
    if (map.has(el)) {
      value = map.get(el)!;
    }
    value.push(text[i]);
    map.set(el, value);
  });

  return map;
}

const generateCategories = async (mappedClusters: Map<number, string[]>) => {
  const categories: string[] = [];
  for (const key of mappedClusters.keys()) {
    const text = mappedClusters.get(key);
    const reviewsAsText = text?.join("\n\n\n");
    try {
      const category = await ai.generate({
        prompt: `
Given the following reviews from a store, give this grouping of reviews a one word category name that describes the category this feedback would reside in.
The one word category must be different than the CURRENT CATEGORIES.

REVIEWS: ${reviewsAsText}
CURRENT CATEGORIES: [${categories.join(",")}]

The output must be only one word. Do not provide an explanation or any other commentary. Just one word.
EXAMPLE OUTPUT: "Cleanliness"
`})
      categories.push(category.text)
      console.log(categories)
    } catch (ex) {
      console.error(ex);
    }
  }
  writeFileSync(MAPPED_RESULT_FILE, JSON.stringify(categories))

}

const main = async () => {
  const oj = readFileSync('out.json', 'utf8')
  const out = JSON.parse(oj)
  const emR = out.map(async (value: string): Promise<Embedding> => {
    let outEmbedding: Embedding = [];
    try {
      outEmbedding = await generateEmbedding(value);
    } catch (ex) {
      console.error(ex);
    }
    return outEmbedding;
  });
  Promise.all(emR).then(async (r) => {
    console.log(r);
    const ans = kmeans(r, CATEGROIES_COUNT, { maxIterations: 10 })
    console.log(ans.clusters);
    const mappedClusters = mapClusters(ans.clusters, out);
    await generateCategories(mappedClusters);
  });
}

main()
