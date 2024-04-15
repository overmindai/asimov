/// few shot generation example:
/// 1. Adds few shot example to an in-memory vector db (HoraDb).
/// 2. Retrieves 3 nearest few shot examples to a query.
/// 3. Use the retrieved examples to generate code for a given prompt using GPT-4.
use asimov::prelude::*;
use futures::StreamExt;
use lazy_static::lazy_static;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug, Clone)]

struct FewShotCodeExample {
    prompt: String,
    code: ExecutableCode,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct ExecutableCode {
    code: String,
}

impl Input for ExecutableCode {
    fn render(&self) -> Result<String> {
        Ok(serde_json::to_string(&self)?)
    }
}

impl Input for FewShotCodeExample {
    fn render(&self) -> Result<String> {
        Ok(format!(
            "Prompt: {}\nCode: {}",
            self.prompt,
            self.code.render()?
        ))
    }
}

impl Embeddable for FewShotCodeExample {
    type Key = String;

    fn key(&self) -> &Self::Key {
        &self.prompt
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY must be set");

    let mut db = HoraDb::new(OpenAiEmbedding::default());
    let namespace = "code_snippets_namespace";
    println!("Created HoraDb instance and set namespace.");

    // Store code snippets in Hora
    db.create_namespace(&namespace).await?;
    println!("Namespace created in HoraDb.");

    db.add_items(&namespace, CODE_SNIPPETS.iter().cloned())
        .await?;
    println!("Added code snippets to the namespace.");

    let query = prompt!(lines! {
        "Generate a python function to calculate the mean of a list of numbers."
    });
    println!("Created query for generating a python function.");

    let k = 3;
    let few_shot_examples = db.knn(&namespace, &query, k).await?;

    println!(
        "Retrieved few-shot examples: {:?}",
        few_shot_examples.render().unwrap()
    );

    let codegen_prompt = prompt!(
        lines! {
            "Please write python code for the requested prompt.",
            "Here are some examples for reference.",
            "{{few_shot_examples}}",
            "Do not add formatting backticks like ```. Just return valid python code.",
            "Prompt: {{query}}",
            "Code:"
        },
        query,
        few_shot_examples
    );
    println!("Created code generation prompt with few-shot examples.");

    let gpt4 = OpenAiLlm::builder()
        .model("gpt-4".to_string())
        .temperature(0.0)
        .build();

    println!("Initialized GPT-4 model.");

    let response: ExecutableCode = gpt4.generate(&codegen_prompt).await?;
    println!("Generated code: {:?}", response.code);

    Ok(())
}

lazy_static! {
    static ref CODE_SNIPPETS: Vec<FewShotCodeExample> = vec![
        FewShotCodeExample {
            prompt: "Generate a python function that calculates the factorial of a number."
                .to_string(),
            code: ExecutableCode {
                code: r#"
def factorial(n):
    if n == 0:
        return 1
    else:
        return n * factorial(n-1)
"#
                .trim()
                .to_string(),
            },
        },
        FewShotCodeExample {
            prompt: "Generate a python function that checks if a number is prime.".to_string(),
            code: ExecutableCode {
                code: r#"
def is_prime(n):
    if n <= 1:
        return False
    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            return False
    return True
"#
                .trim()
                .to_string(),
            },
        },
        FewShotCodeExample {
            prompt: "Generate a python function that reverses a string.".to_string(),
            code: ExecutableCode {
                code: r#"
def reverse_string(s):
    return s[::-1]
"#
                .trim()
                .to_string(),
            },
        },
        FewShotCodeExample {
            prompt: "Generate a python function that checks if a string is a palindrome."
                .to_string(),
            code: ExecutableCode {
                code: r#"
def is_palindrome(s):
    return s == s[::-1]
"#
                .trim()
                .to_string(),
            },
        },
    ];
}
