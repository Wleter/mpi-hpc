use std::{thread::sleep, time::Duration};

use mpi_hpc::distribute;

fn linspace(start: f64, end: f64, n: usize) -> Vec<f64> {
    if n == 1 {
        return vec![start];
    }

    let mut result = Vec::with_capacity(n);
    let step = (end - start) / (n as f64 - 1.0);

    for i in 0..n {
        result.push(start + (i as f64) * step);
    }

    result
}

fn main() {
    distribute!(
        || linspace(0.0, 10.0, 20),
        |x: f64| {
            println!("{x}");
            sleep(Duration::from_secs(1));
            x + 1.0
        },
        |data: Vec<f64>| println!("{:?}", data)
    );
}