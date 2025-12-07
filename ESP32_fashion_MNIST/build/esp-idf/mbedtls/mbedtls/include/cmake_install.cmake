# Install script for directory: D:/LiZhen/Software/esp-idf-v5.4.3/components/mbedtls/mbedtls/include

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "C:/Program Files (x86)/person_detection")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "TRUE")
endif()

# Set path to fallback-tool for dependency-resolution.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "C:/Users/dell/.espressif/tools/xtensa-esp-elf/esp-14.2.0_20250730/xtensa-esp-elf/bin/xtensa-esp32s3-elf-objdump.exe")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/mbedtls" TYPE FILE PERMISSIONS OWNER_READ OWNER_WRITE GROUP_READ WORLD_READ FILES
    "D:/LiZhen/Software/esp-idf-v5.4.3/components/mbedtls/mbedtls/include/mbedtls/aes.h"
    "D:/LiZhen/Software/esp-idf-v5.4.3/components/mbedtls/mbedtls/include/mbedtls/aria.h"
    "D:/LiZhen/Software/esp-idf-v5.4.3/components/mbedtls/mbedtls/include/mbedtls/asn1.h"
    "D:/LiZhen/Software/esp-idf-v5.4.3/components/mbedtls/mbedtls/include/mbedtls/asn1write.h"
    "D:/LiZhen/Software/esp-idf-v5.4.3/components/mbedtls/mbedtls/include/mbedtls/base64.h"
    "D:/LiZhen/Software/esp-idf-v5.4.3/components/mbedtls/mbedtls/include/mbedtls/bignum.h"
    "D:/LiZhen/Software/esp-idf-v5.4.3/components/mbedtls/mbedtls/include/mbedtls/block_cipher.h"
    "D:/LiZhen/Software/esp-idf-v5.4.3/components/mbedtls/mbedtls/include/mbedtls/build_info.h"
    "D:/LiZhen/Software/esp-idf-v5.4.3/components/mbedtls/mbedtls/include/mbedtls/camellia.h"
    "D:/LiZhen/Software/esp-idf-v5.4.3/components/mbedtls/mbedtls/include/mbedtls/ccm.h"
    "D:/LiZhen/Software/esp-idf-v5.4.3/components/mbedtls/mbedtls/include/mbedtls/chacha20.h"
    "D:/LiZhen/Software/esp-idf-v5.4.3/components/mbedtls/mbedtls/include/mbedtls/chachapoly.h"
    "D:/LiZhen/Software/esp-idf-v5.4.3/components/mbedtls/mbedtls/include/mbedtls/check_config.h"
    "D:/LiZhen/Software/esp-idf-v5.4.3/components/mbedtls/mbedtls/include/mbedtls/cipher.h"
    "D:/LiZhen/Software/esp-idf-v5.4.3/components/mbedtls/mbedtls/include/mbedtls/cmac.h"
    "D:/LiZhen/Software/esp-idf-v5.4.3/components/mbedtls/mbedtls/include/mbedtls/compat-2.x.h"
    "D:/LiZhen/Software/esp-idf-v5.4.3/components/mbedtls/mbedtls/include/mbedtls/config_adjust_legacy_crypto.h"
    "D:/LiZhen/Software/esp-idf-v5.4.3/components/mbedtls/mbedtls/include/mbedtls/config_adjust_legacy_from_psa.h"
    "D:/LiZhen/Software/esp-idf-v5.4.3/components/mbedtls/mbedtls/include/mbedtls/config_adjust_psa_from_legacy.h"
    "D:/LiZhen/Software/esp-idf-v5.4.3/components/mbedtls/mbedtls/include/mbedtls/config_adjust_psa_superset_legacy.h"
    "D:/LiZhen/Software/esp-idf-v5.4.3/components/mbedtls/mbedtls/include/mbedtls/config_adjust_ssl.h"
    "D:/LiZhen/Software/esp-idf-v5.4.3/components/mbedtls/mbedtls/include/mbedtls/config_adjust_x509.h"
    "D:/LiZhen/Software/esp-idf-v5.4.3/components/mbedtls/mbedtls/include/mbedtls/config_psa.h"
    "D:/LiZhen/Software/esp-idf-v5.4.3/components/mbedtls/mbedtls/include/mbedtls/constant_time.h"
    "D:/LiZhen/Software/esp-idf-v5.4.3/components/mbedtls/mbedtls/include/mbedtls/ctr_drbg.h"
    "D:/LiZhen/Software/esp-idf-v5.4.3/components/mbedtls/mbedtls/include/mbedtls/debug.h"
    "D:/LiZhen/Software/esp-idf-v5.4.3/components/mbedtls/mbedtls/include/mbedtls/des.h"
    "D:/LiZhen/Software/esp-idf-v5.4.3/components/mbedtls/mbedtls/include/mbedtls/dhm.h"
    "D:/LiZhen/Software/esp-idf-v5.4.3/components/mbedtls/mbedtls/include/mbedtls/ecdh.h"
    "D:/LiZhen/Software/esp-idf-v5.4.3/components/mbedtls/mbedtls/include/mbedtls/ecdsa.h"
    "D:/LiZhen/Software/esp-idf-v5.4.3/components/mbedtls/mbedtls/include/mbedtls/ecjpake.h"
    "D:/LiZhen/Software/esp-idf-v5.4.3/components/mbedtls/mbedtls/include/mbedtls/ecp.h"
    "D:/LiZhen/Software/esp-idf-v5.4.3/components/mbedtls/mbedtls/include/mbedtls/entropy.h"
    "D:/LiZhen/Software/esp-idf-v5.4.3/components/mbedtls/mbedtls/include/mbedtls/error.h"
    "D:/LiZhen/Software/esp-idf-v5.4.3/components/mbedtls/mbedtls/include/mbedtls/gcm.h"
    "D:/LiZhen/Software/esp-idf-v5.4.3/components/mbedtls/mbedtls/include/mbedtls/hkdf.h"
    "D:/LiZhen/Software/esp-idf-v5.4.3/components/mbedtls/mbedtls/include/mbedtls/hmac_drbg.h"
    "D:/LiZhen/Software/esp-idf-v5.4.3/components/mbedtls/mbedtls/include/mbedtls/lms.h"
    "D:/LiZhen/Software/esp-idf-v5.4.3/components/mbedtls/mbedtls/include/mbedtls/mbedtls_config.h"
    "D:/LiZhen/Software/esp-idf-v5.4.3/components/mbedtls/mbedtls/include/mbedtls/md.h"
    "D:/LiZhen/Software/esp-idf-v5.4.3/components/mbedtls/mbedtls/include/mbedtls/md5.h"
    "D:/LiZhen/Software/esp-idf-v5.4.3/components/mbedtls/mbedtls/include/mbedtls/memory_buffer_alloc.h"
    "D:/LiZhen/Software/esp-idf-v5.4.3/components/mbedtls/mbedtls/include/mbedtls/net_sockets.h"
    "D:/LiZhen/Software/esp-idf-v5.4.3/components/mbedtls/mbedtls/include/mbedtls/nist_kw.h"
    "D:/LiZhen/Software/esp-idf-v5.4.3/components/mbedtls/mbedtls/include/mbedtls/oid.h"
    "D:/LiZhen/Software/esp-idf-v5.4.3/components/mbedtls/mbedtls/include/mbedtls/pem.h"
    "D:/LiZhen/Software/esp-idf-v5.4.3/components/mbedtls/mbedtls/include/mbedtls/pk.h"
    "D:/LiZhen/Software/esp-idf-v5.4.3/components/mbedtls/mbedtls/include/mbedtls/pkcs12.h"
    "D:/LiZhen/Software/esp-idf-v5.4.3/components/mbedtls/mbedtls/include/mbedtls/pkcs5.h"
    "D:/LiZhen/Software/esp-idf-v5.4.3/components/mbedtls/mbedtls/include/mbedtls/pkcs7.h"
    "D:/LiZhen/Software/esp-idf-v5.4.3/components/mbedtls/mbedtls/include/mbedtls/platform.h"
    "D:/LiZhen/Software/esp-idf-v5.4.3/components/mbedtls/mbedtls/include/mbedtls/platform_time.h"
    "D:/LiZhen/Software/esp-idf-v5.4.3/components/mbedtls/mbedtls/include/mbedtls/platform_util.h"
    "D:/LiZhen/Software/esp-idf-v5.4.3/components/mbedtls/mbedtls/include/mbedtls/poly1305.h"
    "D:/LiZhen/Software/esp-idf-v5.4.3/components/mbedtls/mbedtls/include/mbedtls/private_access.h"
    "D:/LiZhen/Software/esp-idf-v5.4.3/components/mbedtls/mbedtls/include/mbedtls/psa_util.h"
    "D:/LiZhen/Software/esp-idf-v5.4.3/components/mbedtls/mbedtls/include/mbedtls/ripemd160.h"
    "D:/LiZhen/Software/esp-idf-v5.4.3/components/mbedtls/mbedtls/include/mbedtls/rsa.h"
    "D:/LiZhen/Software/esp-idf-v5.4.3/components/mbedtls/mbedtls/include/mbedtls/sha1.h"
    "D:/LiZhen/Software/esp-idf-v5.4.3/components/mbedtls/mbedtls/include/mbedtls/sha256.h"
    "D:/LiZhen/Software/esp-idf-v5.4.3/components/mbedtls/mbedtls/include/mbedtls/sha3.h"
    "D:/LiZhen/Software/esp-idf-v5.4.3/components/mbedtls/mbedtls/include/mbedtls/sha512.h"
    "D:/LiZhen/Software/esp-idf-v5.4.3/components/mbedtls/mbedtls/include/mbedtls/ssl.h"
    "D:/LiZhen/Software/esp-idf-v5.4.3/components/mbedtls/mbedtls/include/mbedtls/ssl_cache.h"
    "D:/LiZhen/Software/esp-idf-v5.4.3/components/mbedtls/mbedtls/include/mbedtls/ssl_ciphersuites.h"
    "D:/LiZhen/Software/esp-idf-v5.4.3/components/mbedtls/mbedtls/include/mbedtls/ssl_cookie.h"
    "D:/LiZhen/Software/esp-idf-v5.4.3/components/mbedtls/mbedtls/include/mbedtls/ssl_ticket.h"
    "D:/LiZhen/Software/esp-idf-v5.4.3/components/mbedtls/mbedtls/include/mbedtls/threading.h"
    "D:/LiZhen/Software/esp-idf-v5.4.3/components/mbedtls/mbedtls/include/mbedtls/timing.h"
    "D:/LiZhen/Software/esp-idf-v5.4.3/components/mbedtls/mbedtls/include/mbedtls/version.h"
    "D:/LiZhen/Software/esp-idf-v5.4.3/components/mbedtls/mbedtls/include/mbedtls/x509.h"
    "D:/LiZhen/Software/esp-idf-v5.4.3/components/mbedtls/mbedtls/include/mbedtls/x509_crl.h"
    "D:/LiZhen/Software/esp-idf-v5.4.3/components/mbedtls/mbedtls/include/mbedtls/x509_crt.h"
    "D:/LiZhen/Software/esp-idf-v5.4.3/components/mbedtls/mbedtls/include/mbedtls/x509_csr.h"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/psa" TYPE FILE PERMISSIONS OWNER_READ OWNER_WRITE GROUP_READ WORLD_READ FILES
    "D:/LiZhen/Software/esp-idf-v5.4.3/components/mbedtls/mbedtls/include/psa/build_info.h"
    "D:/LiZhen/Software/esp-idf-v5.4.3/components/mbedtls/mbedtls/include/psa/crypto.h"
    "D:/LiZhen/Software/esp-idf-v5.4.3/components/mbedtls/mbedtls/include/psa/crypto_adjust_auto_enabled.h"
    "D:/LiZhen/Software/esp-idf-v5.4.3/components/mbedtls/mbedtls/include/psa/crypto_adjust_config_dependencies.h"
    "D:/LiZhen/Software/esp-idf-v5.4.3/components/mbedtls/mbedtls/include/psa/crypto_adjust_config_key_pair_types.h"
    "D:/LiZhen/Software/esp-idf-v5.4.3/components/mbedtls/mbedtls/include/psa/crypto_adjust_config_synonyms.h"
    "D:/LiZhen/Software/esp-idf-v5.4.3/components/mbedtls/mbedtls/include/psa/crypto_builtin_composites.h"
    "D:/LiZhen/Software/esp-idf-v5.4.3/components/mbedtls/mbedtls/include/psa/crypto_builtin_key_derivation.h"
    "D:/LiZhen/Software/esp-idf-v5.4.3/components/mbedtls/mbedtls/include/psa/crypto_builtin_primitives.h"
    "D:/LiZhen/Software/esp-idf-v5.4.3/components/mbedtls/mbedtls/include/psa/crypto_compat.h"
    "D:/LiZhen/Software/esp-idf-v5.4.3/components/mbedtls/mbedtls/include/psa/crypto_config.h"
    "D:/LiZhen/Software/esp-idf-v5.4.3/components/mbedtls/mbedtls/include/psa/crypto_driver_common.h"
    "D:/LiZhen/Software/esp-idf-v5.4.3/components/mbedtls/mbedtls/include/psa/crypto_driver_contexts_composites.h"
    "D:/LiZhen/Software/esp-idf-v5.4.3/components/mbedtls/mbedtls/include/psa/crypto_driver_contexts_key_derivation.h"
    "D:/LiZhen/Software/esp-idf-v5.4.3/components/mbedtls/mbedtls/include/psa/crypto_driver_contexts_primitives.h"
    "D:/LiZhen/Software/esp-idf-v5.4.3/components/mbedtls/mbedtls/include/psa/crypto_extra.h"
    "D:/LiZhen/Software/esp-idf-v5.4.3/components/mbedtls/mbedtls/include/psa/crypto_legacy.h"
    "D:/LiZhen/Software/esp-idf-v5.4.3/components/mbedtls/mbedtls/include/psa/crypto_platform.h"
    "D:/LiZhen/Software/esp-idf-v5.4.3/components/mbedtls/mbedtls/include/psa/crypto_se_driver.h"
    "D:/LiZhen/Software/esp-idf-v5.4.3/components/mbedtls/mbedtls/include/psa/crypto_sizes.h"
    "D:/LiZhen/Software/esp-idf-v5.4.3/components/mbedtls/mbedtls/include/psa/crypto_struct.h"
    "D:/LiZhen/Software/esp-idf-v5.4.3/components/mbedtls/mbedtls/include/psa/crypto_types.h"
    "D:/LiZhen/Software/esp-idf-v5.4.3/components/mbedtls/mbedtls/include/psa/crypto_values.h"
    )
endif()

