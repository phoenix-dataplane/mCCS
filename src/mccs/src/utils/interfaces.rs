use atoi::atoi;
use nix::sys::socket::{AddressFamily, SockaddrLike};
use socket2::SockAddr;
use thiserror::Error;

#[derive(Debug, Clone, Error)]
pub enum NetInterfaceError {
    #[error("Failed to parse prefix list: {0}")]
    ParsePrefix(String),
    #[error("Nix error: {0}")]
    Nix(#[from] nix::Error),
}

pub struct NetInterfaceSpec {
    pub prefix: String,
    pub port: Option<u16>,
}

pub fn parse_prefix_list(prefix_list: &str) -> Result<Vec<NetInterfaceSpec>, NetInterfaceError> {
    if !prefix_list.is_ascii() {
        Err(NetInterfaceError::ParsePrefix(prefix_list.to_string()))?;
    }
    let mut if_specs = Vec::new();
    let mut pos = 0;
    let bytes = prefix_list.as_bytes();
    let mut curr_if_prefix = String::new();
    while pos < bytes.len() {
        let c = bytes[pos] as char;
        if c == ':' {
            if curr_if_prefix.len() > 0 {
                if pos == bytes.len() - 1 {
                    return Err(NetInterfaceError::ParsePrefix(prefix_list.to_string()));
                }
                let port = atoi::<u16>(&bytes[pos + 1..])
                    .ok_or_else(|| NetInterfaceError::ParsePrefix(prefix_list.to_string()))?;
                let spec = NetInterfaceSpec {
                    prefix: std::mem::take(&mut curr_if_prefix),
                    port: Some(port),
                };
                if_specs.push(spec);
            }
            while pos < bytes.len() && bytes[pos] as char != ',' {
                pos += 1;
            }
        } else if c == ',' || pos == bytes.len() - 1 {
            if c != ',' {
                curr_if_prefix.push(c);
            }
            if curr_if_prefix.len() > 0 {
                let spec = NetInterfaceSpec {
                    prefix: std::mem::take(&mut curr_if_prefix),
                    port: None,
                };
                if_specs.push(spec);
            }
        } else {
            curr_if_prefix.push(c);
        }
        pos += 1;
    }
    Ok(if_specs)
}

pub fn match_interface_list(
    string: &str,
    port: Option<u16>,
    specs: &[NetInterfaceSpec],
    match_exact: bool,
) -> bool {
    if specs.is_empty() {
        return true;
    }
    for spec in specs {
        if spec.port.is_some() && port.is_some() && spec.port.unwrap() != port.unwrap() {
            continue;
        }
        if match_exact {
            if spec.prefix == string {
                return true;
            }
        } else {
            if string.starts_with(&spec.prefix) {
                return true;
            }
        }
    }
    false
}

fn find_interfaces_with_prefix(
    mut prefix_list: &str,
    sock_family: Option<AddressFamily>,
    max_num_interfaces: usize,
) -> Result<Vec<(String, SockAddr)>, NetInterfaceError> {
    if !prefix_list.is_ascii() {
        Err(NetInterfaceError::ParsePrefix(prefix_list.to_string()))?;
    }
    let search_not = prefix_list.chars().nth(0) == Some('^');
    if search_not {
        prefix_list = &prefix_list[1..];
    }
    let search_exact = prefix_list.chars().nth(0) == Some('=');
    if search_exact {
        prefix_list = &prefix_list[1..];
    }
    let specs = parse_prefix_list(prefix_list)?;
    let mut interfaces = Vec::new();
    for interface in nix::ifaddrs::getifaddrs()? {
        if let Some(addr) = interface.address {
            let sockaddr_ptr = addr.as_ptr();
            let sockaddr_len = addr.len();
            let sock_addr = unsafe {
                let (_, sock_addr) = SockAddr::try_init(|sockaddr, len| {
                    std::ptr::copy_nonoverlapping(
                        sockaddr_ptr as *const u8,
                        sockaddr as *mut u8,
                        sockaddr_len as usize,
                    );
                    *len = sockaddr_len;
                    Ok(())
                })
                .unwrap();
                sock_addr
            };
            let family = sock_addr.family();
            if family != AddressFamily::Inet as u16 && family != AddressFamily::Inet6 as u16 {
                continue;
            }

            if let Some(sock_family) = sock_family {
                if family != sock_family as u16 {
                    continue;
                }
            }

            log::trace!(
                "Found interface {}:{:?}",
                interface.interface_name,
                sock_addr
            );

            if family == AddressFamily::Inet6 as u16 {
                let sa = sock_addr.as_socket_ipv6().unwrap();
                if sa.ip().is_loopback() {
                    continue;
                }
            }

            let if_name = interface.interface_name.as_str();
            if !(match_interface_list(if_name, None, &specs, search_exact) ^ search_not) {
                continue;
            }
            let duplicate = interfaces.iter().any(|(name, _)| name == if_name);
            if !duplicate {
                interfaces.push((interface.interface_name, sock_addr));
                if interfaces.len() >= max_num_interfaces {
                    break;
                }
            }
        }
    }
    Ok(interfaces)
}

pub fn find_interfaces(
    specified_prefix: Option<&str>,
    specified_family: Option<AddressFamily>,
    max_num_interfaces: usize,
) -> Result<Vec<(String, SockAddr)>, NetInterfaceError> {
    if let Some(prefix_list) = specified_prefix {
        return find_interfaces_with_prefix(prefix_list, specified_family, max_num_interfaces);
    } else {
        let interfaces = find_interfaces_with_prefix("ib", specified_family, max_num_interfaces)?;
        if !interfaces.is_empty() {
            return Ok(interfaces);
        }
        let interfaces =
            find_interfaces_with_prefix("^docker,lo", specified_family, max_num_interfaces)?;
        if !interfaces.is_empty() {
            return Ok(interfaces);
        }
        let interfaces =
            find_interfaces_with_prefix("docker", specified_family, max_num_interfaces)?;
        if !interfaces.is_empty() {
            return Ok(interfaces);
        }
        return find_interfaces_with_prefix("lo", specified_family, max_num_interfaces);
    }
}
