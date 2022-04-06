#!/bin/bash
set +x
export CUDA_VISIBLE_DEVICES=${1}

itr_array=(100 101 102 103 104 105)
# find "/data1/whj/facenet/datasets/lfw/lfw_mtcnnpy_160" -type f -printf "%h\0" | xargs -0 -L1 basename | sort | uniq -c | sort -k1 -n
dir=$(ls images/)
for i in $dir
do
  for its in ${itr_array[@]}
  do
    ./face_attack_ada.py \
      --data "/data1/whj/facenet/datasets/lfw/lfw_mtcnnpy_160" \
      --itr $its \
      --attack images/$i \
      --target Ian_Thorpe \
      --output $i-to-Ian_Thorpe_`expr $its - 100`.png
  done
done

for i in $dir
do
  ./face_attack.py \
    --data "/data1/whj/facenet/datasets/lfw/lfw_mtcnnpy_160" \
    --attack images/$i \
    --target Ian_Thorpe \
    --output $i-to-Ian_Thorpe_baseline.png
done
    #  10 Bill_McBride
    #  10 Ian_Thorpe
    #  10 Jacques_Rogge
    #  10 Jason_Kidd
    #  10 Javier_Solana
    #  10 Jean-David_Levitte
    #  10 Mohammad_Khatami
    #  10 Muhammad_Ali
    #  10 Paradorn_Srichaphan
    #  10 Paul_Wolfowitz
    #  10 Richard_Gere
    #  10 Tom_Cruise
    #  10 Tom_Hanks
    #  10 Tommy_Thompson
    #  10 Walter_Mondale
    #  11 Ann_Veneman
    #  11 Catherine_Zeta-Jones
    #  11 Condoleezza_Rice
    #  11 James_Kelly
    #  11 Jiri_Novak
    #  11 John_Allen_Muhammad
    #  11 John_Paul_II
    #  11 Kim_Ryong-sung
    #  11 Mark_Philippoussis
    #  11 Mike_Weir
    #  11 Nicanor_Duarte_Frutos
    #  11 Paul_Burrell
    #  11 Richard_Gephardt
    #  11 Sergey_Lavrov
    #  11 Sergio_Vieira_De_Mello
    #  11 Tang_Jiaxuan
    #  12 Adrien_Brody
    #  12 Anna_Kournikova
    #  12 Gonzalo_Sanchez_de_Lozada
    #  12 Harrison_Ford
    #  12 Howard_Dean
    #  12 Jeb_Bush
    #  12 Jennifer_Garner
    #  12 Keanu_Reeves
    #  12 Michael_Jackson
    #  12 Rubens_Barrichello
    #  13 Ari_Fleischer
    #  13 Charles_Moose
    #  13 Edmund_Stoiber
    #  13 George_HW_Bush
    #  13 Gordon_Brown
    #  13 Jackie_Chan
    #  13 Joe_Lieberman
    #  13 Lucio_Gutierrez
    #  13 Queen_Elizabeth_II
    #  13 Salma_Hayek
    #  13 Wen_Jiabao
    #  14 Britney_Spears
    #  14 David_Nalbandian
    #  14 Dick_Cheney
    #  14 Eduardo_Duhalde
    #  14 Hillary_Clinton
    #  14 James_Blake
    #  14 Kim_Clijsters
    #  14 Mahathir_Mohamad
    #  14 Roger_Federer
    #  14 Yoriko_Kawaguchi
    #  15 Andy_Roddick
    #  15 Bill_Simon
    #  15 Dominique_de_Villepin
    #  15 Hu_Jintao
    #  15 Julie_Gerberding
    #  15 Meryl_Streep
    #  15 Mohammed_Al-Douri
    #  15 Nancy_Pelosi
    #  15 Norah_Jones
    #  15 Pierce_Brosnan
    #  15 Taha_Yassin_Ramadan
    #  16 Halle_Berry
    #  16 Tommy_Franks
    #  16 Trent_Lott
    #  17 Bill_Gates
    #  17 Jean_Charest
    #  17 John_Bolton
    #  17 John_Kerry
    #  17 John_Snow
    #  17 Renee_Zellweger
    #  17 Spencer_Abraham
    #  17 Venus_Williams
    #  18 Fidel_Castro
    #  18 Lance_Armstrong
    #  18 Michael_Schumacher
    #  18 Pervez_Musharraf
    #  18 Richard_Myers
    #  19 Abdullah_Gul
    #  19 Carlos_Moya
    #  19 John_Howard
    #  19 Joschka_Fischer
    #  19 Julianne_Moore
    #  19 Nicole_Kidman
    #  19 Tim_Henman
    #  20 Angelina_Jolie
    #  20 Igor_Ivanov
    #  20 Jiang_Zemin
    #  20 Michael_Bloomberg
    #  20 Paul_Bremer
    #  21 Amelie_Mauresmo
    #  21 Carlos_Menem
    #  21 Jennifer_Aniston
    #  21 Jennifer_Lopez
    #  22 George_Robertson
    #  22 Hamid_Karzai
    #  22 Lindsay_Davenport
    #  22 Naomi_Watts
    #  22 Pete_Sampras
    #  23 Jose_Maria_Aznar
    #  23 Saddam_Hussein
    #  23 Tiger_Woods
    #  24 Atal_Bihari_Vajpayee
    #  24 Jeremy_Greenstock
    #  24 Winona_Ryder
    #  25 Tom_Daschle
    #  26 Gray_Davis
    #  26 Rudolph_Giuliani
    #  27 Ricardo_Lagos
    #  28 Jack_Straw
    #  28 Juan_Carlos_Ferrero
    #  29 Bill_Clinton
    #  29 Mahmoud_Abbas
    #  30 Guillermo_Coria
    #  30 Recep_Tayyip_Erdogan
    #  31 David_Beckham
    #  31 John_Negroponte
    #  32 Kofi_Annan
    #  32 Roh_Moo-hyun
    #  32 Vicente_Fox
    #  33 Megawati_Sukarnoputri
    #  33 Silvio_Berlusconi
    #  33 Tom_Ridge
    #  35 Alvaro_Uribe
    #  36 Andre_Agassi
    #  37 Nestor_Kirchner
    #  39 Alejandro_Toledo
    #  39 Hans_Blix
    #  41 Laura_Bush
    #  41 Lleyton_Hewitt
    #  42 Arnold_Schwarzenegger
    #  42 Jennifer_Capriati
    #  44 Gloria_Macapagal_Arroyo
    #  48 Luiz_Inacio_Lula_da_Silva
    #  49 Vladimir_Putin
    #  52 Jacques_Chirac
    #  52 Serena_Williams
    #  53 John_Ashcroft
    #  55 Jean_Chretien
    #  60 Junichiro_Koizumi
    #  71 Hugo_Chavez
    #  77 Ariel_Sharon
    # 109 Gerhard_Schroeder
    # 121 Donald_Rumsfeld
    # 144 Tony_Blair
    # 236 Colin_Powell
    # 530 George_W_Bush