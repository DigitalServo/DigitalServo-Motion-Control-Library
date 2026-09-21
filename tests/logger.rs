use std::time::Instant;
use serde::Serialize;

use dsmc::logger::{DataStorage, serializer};

#[cfg(test)]
#[derive(Serialize)]
struct TestStruct {
    #[serde(serialize_with = "serializer::serialize_int::<4, usize, _>")]
    pub id: usize,
    #[serde(serialize_with = "serializer::serialize_float::<4, f64, _>")]
    pub time: f64,
    #[serde(serialize_with = "serializer::serialize_float_exp::<3, f64, _>")]
    pub value: f64,
}

#[test]
fn test_data_storage() {
    let mut storage = DataStorage::new("./out/out.csv", ',', false).unwrap();

    let timer = Instant::now();

    for id in 0..100 {
        let time = timer.elapsed().as_micros() as f64 * 1e-6;
        let value = (2.0 * std::f64::consts::PI * 10.0 * time).sin();
        let data = TestStruct { id, time, value };
        storage.add(&data).unwrap();

        std::thread::sleep(std::time::Duration::from_millis(1));
    }

    storage.close().unwrap()
}
