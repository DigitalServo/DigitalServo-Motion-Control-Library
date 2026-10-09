use dsmc::logger::DataStorage;

#[test]
fn test_data_storage_separator_header() {
    let path = "./out/logger_separator.csv";
    let mut storage = DataStorage::new(path).unwrap().set_separator(';').set_header(["t", "y"]);
    storage.add(&[0.0, 1.0]).unwrap();
    storage.add(&[0.5, 2.0]).unwrap();
    storage.close().unwrap();

    assert_eq!(std::fs::read_to_string(path).unwrap(), "t;y\n0.0;1.0\n0.5;2.0\n");
}

#[test]
fn test_data_storage_string_header() {
    let names: Vec<String> = vec!["a".into(), "b".into()];
    let headers: Vec<(&str, DataStorage)> = vec![
        ("array", DataStorage::new("./out/header_array.csv").unwrap().set_header(["a", "b"])),
        ("str_slice", DataStorage::new("./out/header_str_slice.csv").unwrap().set_header(&["a", "b"][..])),
        ("vec_ref", DataStorage::new("./out/header_vec_ref.csv").unwrap().set_header(&names)),
        ("vec", DataStorage::new("./out/header_vec.csv").unwrap().set_header(names.clone())),
        ("iter", DataStorage::new("./out/header_iter.csv").unwrap().set_header(["a", "b"].iter().map(|s| s.to_string()))),
    ];

    for (name, mut storage) in headers {
        storage.add(&[1, 2]).unwrap();
        storage.close().unwrap();
        let path = format!("./out/header_{}.csv", name);
        assert_eq!(std::fs::read_to_string(path).unwrap(), "a,b\n1,2\n", "{}", name);
    }
}
