{
  description = "Rust flake";
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-23.11"; # or whatever vers
		rust-overlay.url = "github:oxalica/rust-overlay";
		rust-overlay.inputs.nixpkgs.follows = "nixpkgs";
		flake-utils.url = "github:numtide/flake-utils";
  };
  
  outputs = { self, nixpkgs, rust-overlay, flake-utils, ... }@inputs:
  flake-utils.lib.eachDefaultSystem (system:
		let
			overlays = [ (import rust-overlay) ];
			pkgs = import nixpkgs {
				inherit system overlays;
			};
		in {
			devShells.default = with pkgs; mkShell {
	      buildInputs = [
	        openssl
	        pkg-config
	        # eza
	        # fd
	        (rust-bin.stable.latest.default.override {
						extensions = [ "rust-docs" "rustfmt" "clippy" "rust-src" "rust-analyzer" ];
						targets = [ "wasm32-unknown-unknown" "x86_64-unknown-linux-gnu"];
					})
	      ];

	      # shellHook = ''
	      #   alias ls=eza
	      #   alias find=fd
	      # '';
	    };
		}
	);
}
