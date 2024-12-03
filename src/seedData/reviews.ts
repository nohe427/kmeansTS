import { genkit } from 'genkit';
import vertexAI, { gemini15Flash } from "@genkit-ai/vertexai";
import { writeFileSync } from 'fs';

const GENERATIONS = 25;
const OUTFILE = 'out.json';

const ai = genkit({
  plugins: [
    vertexAI({ location: 'us-central1' }),
  ],
  model: gemini15Flash
})

const main = async () => {
  const output: string[] = [];
  const cat = ["store ambiance", "prices", "staff helpfulness", "service speed", "cleaniness", "parking", "product availability"];

  const r = cat.map(async (category) => {
    for (let i = 0; i < GENERATIONS; i++) {
      try {
        const { text } = await ai.generate({
          prompt: `
You are a customer that is shopping at a pet store called 'Unleashed Potential'.
You are highly annoyed about something in the store and are going to leave feedback on ${category}.
Make sure that your feedback is target specifically on ${category}.
    `,
          config: {
            temperature: 1.5,
          }
        });
        output.push(text);
        console.log(text);
      } catch (ex) {
        console.error(ex);
      }
    }
    Promise.all(r).then(() => {
      writeFileSync(OUTFILE, JSON.stringify(output));
    })
  });
}

main();
