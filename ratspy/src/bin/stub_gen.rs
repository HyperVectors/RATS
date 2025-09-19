use pyo3_stub_gen::Result;

fn main() -> Result<()> {
    // `stub_info` is a function defined by `define_stub_info_gatherer!` macro.
    let stub = ratspy::stub_info()?;
    stub.generate()?;
    std::fs::rename("ratspy.pyi", "ratspy/ratspy.pyi")?;
    Ok(())
}