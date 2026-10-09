//! A transfer function written with `tf!`, and the step response of a seventh-order lag from its
//! partial fraction expansion, printed.
//!
//! `cargo run --example transfer_function`

fn tf_macro() {
    use dsmc::tf;
    let w = 2.0;
    let tf = tf!("{w}^2 /(2s (s^2 + {w}^2))");
    println!("{}", tf);
}

fn step_response() {
    use dsmc::tf;
    let tf = tf!("0.3^7/(s+ 0.3)^7");
    let tf_step = tf!("1 / s");
    let tf = tf * tf_step;

    let x = tf.partial_fraction();
    let y = x.time_response(0.0);
    println!("{}", y);
}

fn main() {
    tf_macro();
    step_response();
}
