use std::collections::HashMap;

use im_rc::Vector;
use nrs_language_server::hex_parsing::{parse};

fn main() {
    let source = include_str!("test.nrs");
    // let source = r#"
    // test
    // println!("{:?}", &source);
    let (ast, errors, semantic_tokens) = parse(source);
    println!("{:?}", errors);
    if let Some(ref ast) = ast {
        println!("{:#?}", ast);
    } else {
        println!("{:?}", errors);
    }
    println!("{:?}", semantic_tokens);
}
