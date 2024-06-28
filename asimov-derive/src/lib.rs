extern crate proc_macro;
use darling::Error;
use darling::{ast::NestedMeta, FromMeta};
use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, ItemStruct};

#[derive(Debug, FromMeta)]
struct AsimovMacroAttributes {
    #[darling(default)]
    key: Option<syn::Ident>,
}

#[proc_macro_attribute]
pub fn asimov(attr: TokenStream, item: TokenStream) -> TokenStream {
    let input = parse_macro_input!(item as ItemStruct);
    let struct_name = &input.ident;

    let attr_list = match NestedMeta::parse_meta_list(attr.into()) {
        Ok(v) => v,
        Err(e) => {
            return TokenStream::from(Error::from(e).write_errors());
        }
    };

    let asimov_attr = match AsimovMacroAttributes::from_list(&attr_list) {
        Ok(v) => v,
        Err(e) => {
            return TokenStream::from(e.write_errors());
        }
    };

    let render_impl = input.fields.iter().map(|f| {
        let name = &f.ident.clone().expect("Field without a name");
        quote! {
            {
                // Directly call `render` trusting that the correct implementation will be used
                // This could be a custom implementation or the default one for types that implement `Display`
                let result: Result<String, AsimovError> = self.#name.render();
                result
            }
        }
    });

    let embeddable_impl = if let Some(key) = asimov_attr.key.clone() {
        let key_type = input
            .fields
            .iter()
            .find_map(|f| {
                if f.ident.as_ref() == Some(&key) {
                    Some(f.ty.clone())
                } else {
                    None
                }
            })
            .expect("Key field not found in struct");

        quote! {
            impl Embeddable for #struct_name {
                type Key = #key_type;

                fn key(&self) -> Self::Key {
                    self.#key.clone()
                }
            }
        }
    } else {
        quote! {
            impl Embeddable for #struct_name {
                type Key = Self;

                fn key(&self) -> &Self::Key {
                    &self
                }
            }
        }
    };

    let tok_stream = quote! {
        #input

        impl Input for #struct_name {
            fn render(&self) -> Result<String, AsimovError> {
                let fields_rendered: Result<Vec<String>, AsimovError> = vec![#(#render_impl),*].into_iter().collect();
                fields_rendered.map(|v| v.join(", "))
            }
        }

        #embeddable_impl
    };

    tok_stream.into()
}
