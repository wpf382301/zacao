/*
Navicat MySQL Data Transfer

Source Server         : localhost_3306
Source Server Version : 50721
Source Host           : localhost:3306
Source Database       : zacao

Target Server Type    : MYSQL
Target Server Version : 50721
File Encoding         : 65001

Date: 2018-04-13 09:16:17
*/

SET FOREIGN_KEY_CHECKS=0;

-- ----------------------------
-- Table structure for `hash_3conv_128_128`
-- ----------------------------
DROP TABLE IF EXISTS `hash_3conv_128_128`;
CREATE TABLE `hash_3conv_128_128` (
`id`  int(11) NOT NULL AUTO_INCREMENT ,
`long_hash_code`  text CHARACTER SET utf8 COLLATE utf8_general_ci NOT NULL ,
`short_hash_code`  text CHARACTER SET utf8 COLLATE utf8_general_ci NOT NULL ,
`directory`  varchar(255) CHARACTER SET utf8 COLLATE utf8_general_ci NOT NULL ,
`label`  varchar(255) CHARACTER SET utf8 COLLATE utf8_general_ci NOT NULL ,
PRIMARY KEY (`id`)
)
ENGINE=InnoDB
DEFAULT CHARACTER SET=utf8 COLLATE=utf8_general_ci
AUTO_INCREMENT=5001

;

-- ----------------------------
-- Table structure for `hash_3conv_128_48`
-- ----------------------------
DROP TABLE IF EXISTS `hash_3conv_128_48`;
CREATE TABLE `hash_3conv_128_48` (
`id`  int(11) NOT NULL AUTO_INCREMENT ,
`long_hash_code`  text CHARACTER SET utf8 COLLATE utf8_general_ci NOT NULL ,
`short_hash_code`  text CHARACTER SET utf8 COLLATE utf8_general_ci NOT NULL ,
`directory`  varchar(255) CHARACTER SET utf8 COLLATE utf8_general_ci NOT NULL ,
`label`  varchar(255) CHARACTER SET utf8 COLLATE utf8_general_ci NOT NULL ,
PRIMARY KEY (`id`)
)
ENGINE=InnoDB
DEFAULT CHARACTER SET=utf8 COLLATE=utf8_general_ci
AUTO_INCREMENT=5001

;

-- ----------------------------
-- Table structure for `hash_3conv_128_64`
-- ----------------------------
DROP TABLE IF EXISTS `hash_3conv_128_64`;
CREATE TABLE `hash_3conv_128_64` (
`id`  int(11) NOT NULL AUTO_INCREMENT ,
`long_hash_code`  text CHARACTER SET utf8 COLLATE utf8_general_ci NOT NULL ,
`short_hash_code`  text CHARACTER SET utf8 COLLATE utf8_general_ci NOT NULL ,
`directory`  varchar(255) CHARACTER SET utf8 COLLATE utf8_general_ci NOT NULL ,
`label`  varchar(255) CHARACTER SET utf8 COLLATE utf8_general_ci NOT NULL ,
PRIMARY KEY (`id`)
)
ENGINE=InnoDB
DEFAULT CHARACTER SET=utf8 COLLATE=utf8_general_ci
AUTO_INCREMENT=5001

;

-- ----------------------------
-- Table structure for `hash_3conv_192_48`
-- ----------------------------
DROP TABLE IF EXISTS `hash_3conv_192_48`;
CREATE TABLE `hash_3conv_192_48` (
`id`  int(11) NOT NULL AUTO_INCREMENT ,
`long_hash_code`  text CHARACTER SET utf8 COLLATE utf8_general_ci NOT NULL ,
`short_hash_code`  varchar(255) CHARACTER SET utf8 COLLATE utf8_general_ci NOT NULL ,
`directory`  varchar(255) CHARACTER SET utf8 COLLATE utf8_general_ci NOT NULL ,
`label`  varchar(255) CHARACTER SET utf8 COLLATE utf8_general_ci NOT NULL ,
PRIMARY KEY (`id`)
)
ENGINE=InnoDB
DEFAULT CHARACTER SET=utf8 COLLATE=utf8_general_ci
AUTO_INCREMENT=5001

;

-- ----------------------------
-- Table structure for `hash_3conv_256_128`
-- ----------------------------
DROP TABLE IF EXISTS `hash_3conv_256_128`;
CREATE TABLE `hash_3conv_256_128` (
`id`  int(11) NOT NULL AUTO_INCREMENT ,
`long_hash_code`  text CHARACTER SET utf8 COLLATE utf8_general_ci NOT NULL ,
`short_hash_code`  text CHARACTER SET utf8 COLLATE utf8_general_ci NOT NULL ,
`directory`  varchar(255) CHARACTER SET utf8 COLLATE utf8_general_ci NOT NULL ,
`label`  varchar(255) CHARACTER SET utf8 COLLATE utf8_general_ci NOT NULL ,
PRIMARY KEY (`id`)
)
ENGINE=InnoDB
DEFAULT CHARACTER SET=utf8 COLLATE=utf8_general_ci
AUTO_INCREMENT=5001

;

-- ----------------------------
-- Table structure for `hash_3conv_256_48`
-- ----------------------------
DROP TABLE IF EXISTS `hash_3conv_256_48`;
CREATE TABLE `hash_3conv_256_48` (
`id`  int(11) NOT NULL AUTO_INCREMENT ,
`long_hash_code`  text CHARACTER SET utf8 COLLATE utf8_general_ci NOT NULL ,
`short_hash_code`  varchar(255) CHARACTER SET utf8 COLLATE utf8_general_ci NOT NULL ,
`directory`  varchar(255) CHARACTER SET utf8 COLLATE utf8_general_ci NOT NULL ,
`label`  varchar(255) CHARACTER SET utf8 COLLATE utf8_general_ci NOT NULL ,
PRIMARY KEY (`id`)
)
ENGINE=InnoDB
DEFAULT CHARACTER SET=utf8 COLLATE=utf8_general_ci
AUTO_INCREMENT=5001

;

-- ----------------------------
-- Table structure for `hash_3conv_256_64`
-- ----------------------------
DROP TABLE IF EXISTS `hash_3conv_256_64`;
CREATE TABLE `hash_3conv_256_64` (
`id`  int(11) NOT NULL AUTO_INCREMENT ,
`long_hash_code`  text CHARACTER SET utf8 COLLATE utf8_general_ci NOT NULL ,
`short_hash_code`  varchar(255) CHARACTER SET utf8 COLLATE utf8_general_ci NOT NULL ,
`directory`  varchar(255) CHARACTER SET utf8 COLLATE utf8_general_ci NOT NULL ,
`label`  varchar(255) CHARACTER SET utf8 COLLATE utf8_general_ci NOT NULL ,
PRIMARY KEY (`id`)
)
ENGINE=InnoDB
DEFAULT CHARACTER SET=utf8 COLLATE=utf8_general_ci
AUTO_INCREMENT=5001

;

-- ----------------------------
-- Table structure for `hash_4conv_128_128`
-- ----------------------------
DROP TABLE IF EXISTS `hash_4conv_128_128`;
CREATE TABLE `hash_4conv_128_128` (
`id`  int(11) NOT NULL AUTO_INCREMENT ,
`long_hash_code`  text CHARACTER SET utf8 COLLATE utf8_general_ci NOT NULL ,
`short_hash_code`  text CHARACTER SET utf8 COLLATE utf8_general_ci NOT NULL ,
`directory`  varchar(255) CHARACTER SET utf8 COLLATE utf8_general_ci NOT NULL ,
`label`  varchar(255) CHARACTER SET utf8 COLLATE utf8_general_ci NOT NULL ,
PRIMARY KEY (`id`)
)
ENGINE=InnoDB
DEFAULT CHARACTER SET=utf8 COLLATE=utf8_general_ci
AUTO_INCREMENT=5001

;

-- ----------------------------
-- Table structure for `hash_4conv_128_48`
-- ----------------------------
DROP TABLE IF EXISTS `hash_4conv_128_48`;
CREATE TABLE `hash_4conv_128_48` (
`id`  int(11) NOT NULL AUTO_INCREMENT ,
`long_hash_code`  text CHARACTER SET utf8 COLLATE utf8_general_ci NOT NULL ,
`short_hash_code`  varchar(255) CHARACTER SET utf8 COLLATE utf8_general_ci NOT NULL ,
`directory`  varchar(255) CHARACTER SET utf8 COLLATE utf8_general_ci NOT NULL ,
`label`  varchar(255) CHARACTER SET utf8 COLLATE utf8_general_ci NOT NULL ,
PRIMARY KEY (`id`)
)
ENGINE=InnoDB
DEFAULT CHARACTER SET=utf8 COLLATE=utf8_general_ci
AUTO_INCREMENT=5001

;

-- ----------------------------
-- Table structure for `hash_4conv_128_64`
-- ----------------------------
DROP TABLE IF EXISTS `hash_4conv_128_64`;
CREATE TABLE `hash_4conv_128_64` (
`id`  int(11) NOT NULL AUTO_INCREMENT ,
`long_hash_code`  text CHARACTER SET utf8 COLLATE utf8_general_ci NOT NULL ,
`short_hash_code`  varchar(255) CHARACTER SET utf8 COLLATE utf8_general_ci NOT NULL ,
`directory`  varchar(255) CHARACTER SET utf8 COLLATE utf8_general_ci NOT NULL ,
`label`  varchar(255) CHARACTER SET utf8 COLLATE utf8_general_ci NOT NULL ,
PRIMARY KEY (`id`)
)
ENGINE=InnoDB
DEFAULT CHARACTER SET=utf8 COLLATE=utf8_general_ci
AUTO_INCREMENT=5001

;

-- ----------------------------
-- Table structure for `hash_4conv_256_128`
-- ----------------------------
DROP TABLE IF EXISTS `hash_4conv_256_128`;
CREATE TABLE `hash_4conv_256_128` (
`id`  int(11) NOT NULL AUTO_INCREMENT ,
`long_hash_code`  text CHARACTER SET utf8 COLLATE utf8_general_ci NOT NULL ,
`short_hash_code`  text CHARACTER SET utf8 COLLATE utf8_general_ci NOT NULL ,
`directory`  varchar(255) CHARACTER SET utf8 COLLATE utf8_general_ci NOT NULL ,
`label`  varchar(255) CHARACTER SET utf8 COLLATE utf8_general_ci NOT NULL ,
PRIMARY KEY (`id`)
)
ENGINE=InnoDB
DEFAULT CHARACTER SET=utf8 COLLATE=utf8_general_ci
AUTO_INCREMENT=5001

;

-- ----------------------------
-- Table structure for `hash_4conv_256_48`
-- ----------------------------
DROP TABLE IF EXISTS `hash_4conv_256_48`;
CREATE TABLE `hash_4conv_256_48` (
`id`  int(11) NOT NULL AUTO_INCREMENT ,
`long_hash_code`  text CHARACTER SET utf8 COLLATE utf8_general_ci NOT NULL ,
`short_hash_code`  varchar(255) CHARACTER SET utf8 COLLATE utf8_general_ci NOT NULL ,
`directory`  varchar(255) CHARACTER SET utf8 COLLATE utf8_general_ci NOT NULL ,
`label`  varchar(255) CHARACTER SET utf8 COLLATE utf8_general_ci NOT NULL ,
PRIMARY KEY (`id`)
)
ENGINE=InnoDB
DEFAULT CHARACTER SET=utf8 COLLATE=utf8_general_ci
AUTO_INCREMENT=5001

;

-- ----------------------------
-- Table structure for `hash_4conv_256_64`
-- ----------------------------
DROP TABLE IF EXISTS `hash_4conv_256_64`;
CREATE TABLE `hash_4conv_256_64` (
`id`  int(11) NOT NULL AUTO_INCREMENT ,
`long_hash_code`  text CHARACTER SET utf8 COLLATE utf8_general_ci NOT NULL ,
`short_hash_code`  varchar(255) CHARACTER SET utf8 COLLATE utf8_general_ci NOT NULL ,
`directory`  varchar(255) CHARACTER SET utf8 COLLATE utf8_general_ci NOT NULL ,
`label`  varchar(255) CHARACTER SET utf8 COLLATE utf8_general_ci NOT NULL ,
PRIMARY KEY (`id`)
)
ENGINE=InnoDB
DEFAULT CHARACTER SET=utf8 COLLATE=utf8_general_ci
AUTO_INCREMENT=5001

;

-- ----------------------------
-- Table structure for `suoyin_3conv_128_128`
-- ----------------------------
DROP TABLE IF EXISTS `suoyin_3conv_128_128`;
CREATE TABLE `suoyin_3conv_128_128` (
`id`  int(11) NOT NULL AUTO_INCREMENT ,
`short_hash_code`  text CHARACTER SET utf8 COLLATE utf8_general_ci NOT NULL ,
PRIMARY KEY (`id`)
)
ENGINE=InnoDB
DEFAULT CHARACTER SET=utf8 COLLATE=utf8_general_ci
AUTO_INCREMENT=3841

;

-- ----------------------------
-- Table structure for `suoyin_3conv_128_48`
-- ----------------------------
DROP TABLE IF EXISTS `suoyin_3conv_128_48`;
CREATE TABLE `suoyin_3conv_128_48` (
`id`  int(11) NOT NULL AUTO_INCREMENT ,
`short_hash_code`  text CHARACTER SET utf8 COLLATE utf8_general_ci NOT NULL ,
PRIMARY KEY (`id`)
)
ENGINE=InnoDB
DEFAULT CHARACTER SET=utf8 COLLATE=utf8_general_ci
AUTO_INCREMENT=1366

;

-- ----------------------------
-- Table structure for `suoyin_3conv_128_64`
-- ----------------------------
DROP TABLE IF EXISTS `suoyin_3conv_128_64`;
CREATE TABLE `suoyin_3conv_128_64` (
`id`  int(11) NOT NULL AUTO_INCREMENT ,
`short_hash_code`  text CHARACTER SET utf8 COLLATE utf8_general_ci NOT NULL ,
PRIMARY KEY (`id`)
)
ENGINE=InnoDB
DEFAULT CHARACTER SET=utf8 COLLATE=utf8_general_ci
AUTO_INCREMENT=2568

;

-- ----------------------------
-- Table structure for `suoyin_3conv_192_48`
-- ----------------------------
DROP TABLE IF EXISTS `suoyin_3conv_192_48`;
CREATE TABLE `suoyin_3conv_192_48` (
`id`  int(11) NOT NULL AUTO_INCREMENT ,
`short_hash_code`  varchar(255) CHARACTER SET utf8 COLLATE utf8_general_ci NOT NULL ,
PRIMARY KEY (`id`),
UNIQUE INDEX `short_hash_code` (`short_hash_code`) USING BTREE 
)
ENGINE=InnoDB
DEFAULT CHARACTER SET=utf8 COLLATE=utf8_general_ci
AUTO_INCREMENT=1464

;

-- ----------------------------
-- Table structure for `suoyin_3conv_256_128`
-- ----------------------------
DROP TABLE IF EXISTS `suoyin_3conv_256_128`;
CREATE TABLE `suoyin_3conv_256_128` (
`id`  int(11) NOT NULL AUTO_INCREMENT ,
`short_hash_code`  text CHARACTER SET utf8 COLLATE utf8_general_ci NOT NULL ,
PRIMARY KEY (`id`)
)
ENGINE=InnoDB
DEFAULT CHARACTER SET=utf8 COLLATE=utf8_general_ci
AUTO_INCREMENT=3954

;

-- ----------------------------
-- Table structure for `suoyin_3conv_256_48`
-- ----------------------------
DROP TABLE IF EXISTS `suoyin_3conv_256_48`;
CREATE TABLE `suoyin_3conv_256_48` (
`id`  int(11) NOT NULL AUTO_INCREMENT ,
`short_hash_code`  varchar(255) CHARACTER SET utf8 COLLATE utf8_general_ci NOT NULL ,
PRIMARY KEY (`id`),
UNIQUE INDEX `short_hash_code` (`short_hash_code`) USING BTREE 
)
ENGINE=InnoDB
DEFAULT CHARACTER SET=utf8 COLLATE=utf8_general_ci
AUTO_INCREMENT=1136

;

-- ----------------------------
-- Table structure for `suoyin_3conv_256_64`
-- ----------------------------
DROP TABLE IF EXISTS `suoyin_3conv_256_64`;
CREATE TABLE `suoyin_3conv_256_64` (
`id`  int(11) NOT NULL AUTO_INCREMENT ,
`short_hash_code`  varchar(255) CHARACTER SET utf8 COLLATE utf8_general_ci NOT NULL ,
PRIMARY KEY (`id`),
UNIQUE INDEX `short_hash_code` (`short_hash_code`) USING BTREE 
)
ENGINE=InnoDB
DEFAULT CHARACTER SET=utf8 COLLATE=utf8_general_ci
AUTO_INCREMENT=1638

;

-- ----------------------------
-- Table structure for `suoyin_4conv_128_128`
-- ----------------------------
DROP TABLE IF EXISTS `suoyin_4conv_128_128`;
CREATE TABLE `suoyin_4conv_128_128` (
`id`  int(11) NOT NULL AUTO_INCREMENT ,
`short_hash_code`  text CHARACTER SET utf8 COLLATE utf8_general_ci NOT NULL ,
PRIMARY KEY (`id`)
)
ENGINE=InnoDB
DEFAULT CHARACTER SET=utf8 COLLATE=utf8_general_ci
AUTO_INCREMENT=7887

;

-- ----------------------------
-- Table structure for `suoyin_4conv_128_48`
-- ----------------------------
DROP TABLE IF EXISTS `suoyin_4conv_128_48`;
CREATE TABLE `suoyin_4conv_128_48` (
`id`  int(11) NOT NULL AUTO_INCREMENT ,
`short_hash_code`  varchar(255) CHARACTER SET utf8 COLLATE utf8_general_ci NOT NULL ,
PRIMARY KEY (`id`),
UNIQUE INDEX `short_hash_code` (`short_hash_code`) USING BTREE 
)
ENGINE=InnoDB
DEFAULT CHARACTER SET=utf8 COLLATE=utf8_general_ci
AUTO_INCREMENT=1110

;

-- ----------------------------
-- Table structure for `suoyin_4conv_128_64`
-- ----------------------------
DROP TABLE IF EXISTS `suoyin_4conv_128_64`;
CREATE TABLE `suoyin_4conv_128_64` (
`id`  int(11) NOT NULL AUTO_INCREMENT ,
`short_hash_code`  varchar(255) CHARACTER SET utf8 COLLATE utf8_general_ci NOT NULL ,
PRIMARY KEY (`id`),
UNIQUE INDEX `short_hash_code` (`short_hash_code`) USING BTREE 
)
ENGINE=InnoDB
DEFAULT CHARACTER SET=utf8 COLLATE=utf8_general_ci
AUTO_INCREMENT=2132

;

-- ----------------------------
-- Table structure for `suoyin_4conv_256_128`
-- ----------------------------
DROP TABLE IF EXISTS `suoyin_4conv_256_128`;
CREATE TABLE `suoyin_4conv_256_128` (
`id`  int(11) NOT NULL AUTO_INCREMENT ,
`short_hash_code`  text CHARACTER SET utf8 COLLATE utf8_general_ci NOT NULL ,
PRIMARY KEY (`id`)
)
ENGINE=InnoDB
DEFAULT CHARACTER SET=utf8 COLLATE=utf8_general_ci
AUTO_INCREMENT=4166

;

-- ----------------------------
-- Table structure for `suoyin_4conv_256_48`
-- ----------------------------
DROP TABLE IF EXISTS `suoyin_4conv_256_48`;
CREATE TABLE `suoyin_4conv_256_48` (
`id`  int(11) NOT NULL AUTO_INCREMENT ,
`short_hash_code`  varchar(255) CHARACTER SET utf8 COLLATE utf8_general_ci NOT NULL ,
PRIMARY KEY (`id`),
UNIQUE INDEX `short_hash_code` (`short_hash_code`) USING BTREE 
)
ENGINE=InnoDB
DEFAULT CHARACTER SET=utf8 COLLATE=utf8_general_ci
AUTO_INCREMENT=1811

;

-- ----------------------------
-- Table structure for `suoyin_4conv_256_64`
-- ----------------------------
DROP TABLE IF EXISTS `suoyin_4conv_256_64`;
CREATE TABLE `suoyin_4conv_256_64` (
`id`  int(11) NOT NULL AUTO_INCREMENT ,
`short_hash_code`  varchar(255) CHARACTER SET utf8 COLLATE utf8_general_ci NOT NULL ,
PRIMARY KEY (`id`),
UNIQUE INDEX `short_hash_code` (`short_hash_code`) USING BTREE 
)
ENGINE=InnoDB
DEFAULT CHARACTER SET=utf8 COLLATE=utf8_general_ci
AUTO_INCREMENT=2402

;

-- ----------------------------
-- Auto increment value for `hash_3conv_128_128`
-- ----------------------------
ALTER TABLE `hash_3conv_128_128` AUTO_INCREMENT=5001;

-- ----------------------------
-- Auto increment value for `hash_3conv_128_48`
-- ----------------------------
ALTER TABLE `hash_3conv_128_48` AUTO_INCREMENT=5001;

-- ----------------------------
-- Auto increment value for `hash_3conv_128_64`
-- ----------------------------
ALTER TABLE `hash_3conv_128_64` AUTO_INCREMENT=5001;

-- ----------------------------
-- Auto increment value for `hash_3conv_192_48`
-- ----------------------------
ALTER TABLE `hash_3conv_192_48` AUTO_INCREMENT=5001;

-- ----------------------------
-- Auto increment value for `hash_3conv_256_128`
-- ----------------------------
ALTER TABLE `hash_3conv_256_128` AUTO_INCREMENT=5001;

-- ----------------------------
-- Auto increment value for `hash_3conv_256_48`
-- ----------------------------
ALTER TABLE `hash_3conv_256_48` AUTO_INCREMENT=5001;

-- ----------------------------
-- Auto increment value for `hash_3conv_256_64`
-- ----------------------------
ALTER TABLE `hash_3conv_256_64` AUTO_INCREMENT=5001;

-- ----------------------------
-- Auto increment value for `hash_4conv_128_128`
-- ----------------------------
ALTER TABLE `hash_4conv_128_128` AUTO_INCREMENT=5001;

-- ----------------------------
-- Auto increment value for `hash_4conv_128_48`
-- ----------------------------
ALTER TABLE `hash_4conv_128_48` AUTO_INCREMENT=5001;

-- ----------------------------
-- Auto increment value for `hash_4conv_128_64`
-- ----------------------------
ALTER TABLE `hash_4conv_128_64` AUTO_INCREMENT=5001;

-- ----------------------------
-- Auto increment value for `hash_4conv_256_128`
-- ----------------------------
ALTER TABLE `hash_4conv_256_128` AUTO_INCREMENT=5001;

-- ----------------------------
-- Auto increment value for `hash_4conv_256_48`
-- ----------------------------
ALTER TABLE `hash_4conv_256_48` AUTO_INCREMENT=5001;

-- ----------------------------
-- Auto increment value for `hash_4conv_256_64`
-- ----------------------------
ALTER TABLE `hash_4conv_256_64` AUTO_INCREMENT=5001;

-- ----------------------------
-- Auto increment value for `suoyin_3conv_128_128`
-- ----------------------------
ALTER TABLE `suoyin_3conv_128_128` AUTO_INCREMENT=3841;

-- ----------------------------
-- Auto increment value for `suoyin_3conv_128_48`
-- ----------------------------
ALTER TABLE `suoyin_3conv_128_48` AUTO_INCREMENT=1366;

-- ----------------------------
-- Auto increment value for `suoyin_3conv_128_64`
-- ----------------------------
ALTER TABLE `suoyin_3conv_128_64` AUTO_INCREMENT=2568;

-- ----------------------------
-- Auto increment value for `suoyin_3conv_192_48`
-- ----------------------------
ALTER TABLE `suoyin_3conv_192_48` AUTO_INCREMENT=1464;

-- ----------------------------
-- Auto increment value for `suoyin_3conv_256_128`
-- ----------------------------
ALTER TABLE `suoyin_3conv_256_128` AUTO_INCREMENT=3954;

-- ----------------------------
-- Auto increment value for `suoyin_3conv_256_48`
-- ----------------------------
ALTER TABLE `suoyin_3conv_256_48` AUTO_INCREMENT=1136;

-- ----------------------------
-- Auto increment value for `suoyin_3conv_256_64`
-- ----------------------------
ALTER TABLE `suoyin_3conv_256_64` AUTO_INCREMENT=1638;

-- ----------------------------
-- Auto increment value for `suoyin_4conv_128_128`
-- ----------------------------
ALTER TABLE `suoyin_4conv_128_128` AUTO_INCREMENT=7887;

-- ----------------------------
-- Auto increment value for `suoyin_4conv_128_48`
-- ----------------------------
ALTER TABLE `suoyin_4conv_128_48` AUTO_INCREMENT=1110;

-- ----------------------------
-- Auto increment value for `suoyin_4conv_128_64`
-- ----------------------------
ALTER TABLE `suoyin_4conv_128_64` AUTO_INCREMENT=2132;

-- ----------------------------
-- Auto increment value for `suoyin_4conv_256_128`
-- ----------------------------
ALTER TABLE `suoyin_4conv_256_128` AUTO_INCREMENT=4166;

-- ----------------------------
-- Auto increment value for `suoyin_4conv_256_48`
-- ----------------------------
ALTER TABLE `suoyin_4conv_256_48` AUTO_INCREMENT=1811;

-- ----------------------------
-- Auto increment value for `suoyin_4conv_256_64`
-- ----------------------------
ALTER TABLE `suoyin_4conv_256_64` AUTO_INCREMENT=2402;
